import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from prompt.prompts import SYSTEM
import json
import asyncio
from datetime import date
from dotenv import load_dotenv

# ── Rate limiting (in-memory, resets daily) ───────────────────────────────────
_chat_limits: Dict[str, Dict] = {}
_tafsir_limits: Dict[str, Dict] = {}
FREE_CHAT_LIMIT = 50
FREE_TAFSIR_LIMIT = 50
_SYSTEM_THREAD_IDS = {'default', 'anonymous', 'title-generator', 'summary-generator'}

def check_and_increment(store: Dict, user_id: str, limit: int) -> bool:
    today = str(date.today())
    if user_id not in store or store[user_id]['date'] != today:
        store[user_id] = {'count': 0, 'date': today}
    if store[user_id]['count'] >= limit:
        return False
    store[user_id]['count'] += 1
    return True

load_dotenv()

# Download FAISS index from Hugging Face if not present locally
def ensure_faiss_index():
    import os
    if os.path.exists("faiss_index_cleaned/index.faiss"):
        return
    print("Downloading FAISS index from Hugging Face...")
    from huggingface_hub import hf_hub_download
    os.makedirs("faiss_index_cleaned", exist_ok=True)
    hf_hub_download(
        repo_id=os.getenv("HF_REPO_ID"),
        filename="index.faiss",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
        local_dir="faiss_index_cleaned",
    )
    hf_hub_download(
        repo_id=os.getenv("HF_REPO_ID"),
        filename="index.pkl",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
        local_dir="faiss_index_cleaned",
    )
    print("FAISS index downloaded!")

ensure_faiss_index()

app = FastAPI(title="Nur Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index_cleaned")
KALIMAT_KEY = os.getenv("KALIMAT_API_KEY", "c16429ab-eec1-4d6e-ad6f-249f1618346c")
KALIMAT_URL = "https://api.kalimat.dev/api/v2/search"

_chatbot = None
_retriever = None
_llm = None
_llm_stream = None

def get_agent():
    global _chatbot, _retriever, _llm, _llm_stream

    if _chatbot is not None:
        return _chatbot, _retriever, _llm, _llm_stream

    print("Loading Islamic AI agent...")

    from langchain_openai import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.tools import tool
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, SystemMessage
    from typing import TypedDict, Annotated
    import requests as http_requests

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    _retriever = vector_store.as_retriever()

    def kalimat_search(query: str, content_type: str):
        response = http_requests.get(
            KALIMAT_URL,
            params={"query": query, "getText": "true", "contentType": content_type, "userLang": "ar,en"},
            headers={"X-API-Key": KALIMAT_KEY},
            timeout=10,
        )
        return response.json()

    @tool
    def jurisprudence_query(query: str):
        """Query the Islamic jurisprudence knowledge base."""
        docs = _retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])

    @tool
    def quran_search(query: str):
        """Search the Quran for relevant verses."""
        return kalimat_search(query, "quran")

    @tool
    def hadith_search(query: str):
        """Search the Hadith collections for relevant narrations."""
        return kalimat_search(query, "sunnah")

    @tool
    def search_azkar(query: str):
        """Search for relevant Azkar and supplications."""
        return kalimat_search(query, "azkar")

    tools = [jurisprudence_query, quran_search, hadith_search, search_azkar]

    # gpt-4o for tool use (accuracy matters)
    _llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    llm_with_tools = _llm.bind_tools(tools)

    # gpt-4o-mini for streaming (fast, cheap, good enough)
    _llm_stream = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)

    class AgentState(TypedDict):
        messages: Annotated[List, add_messages]

    def agent_node(state: AgentState):
        messages = state["messages"]
        full_messages = [SystemMessage(content=SYSTEM)] + messages
        resp = llm_with_tools.invoke(full_messages)
        return {"messages": [resp]}

    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    _chatbot = graph.compile(checkpointer=InMemorySaver())

    print("Islamic AI agent loaded!")
    return _chatbot, _retriever, _llm, _llm_stream


# ── Models ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = "default"
    user_memory: Optional[Dict[str, Any]] = None
    convo_summaries: Optional[List[Dict[str, Any]]] = None
    is_pro: Optional[bool] = False

class VerseExplainRequest(BaseModel):
    surah: str
    ayah: int
    arabic: str
    translation: str
    user_id: Optional[str] = None
    is_pro: Optional[bool] = False

class LessonRequest(BaseModel):
    level: Optional[str] = "intermediate"
    topic: Optional[str] = None
    lang: Optional[str] = "en"

class ReflectionRequest(BaseModel):
    prayer_time: Optional[str] = "general"


# ── Build memory context ──────────────────────────────────────────────────────
def build_memory_context(user_memory=None, convo_summaries=None):
    context = ""
    if user_memory:
        context += "\n\nUSER PROFILE:\n"
        if user_memory.get('name'):
            context += f"- Name: {user_memory['name']}\n"
        if user_memory.get('level'):
            context += f"- Learning level: {user_memory['level']}\n"
        if user_memory.get('interests'):
            context += f"- Interests: {', '.join(user_memory['interests'])}\n"
    if convo_summaries:
        context += "\n\nPAST CONVERSATION SUMMARIES:\n"
        for s in convo_summaries[:3]:
            context += f"- {s.get('summary', '')}\n"
    return context


# ── Streaming: full conversation history ──────────────────────────────────────
async def stream_agent_response(messages: List[Message], thread_id: str, user_memory=None, convo_summaries=None):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

    chatbot, _, _, llm_stream = get_agent()
    memory_context = build_memory_context(user_memory, convo_summaries)
    system_with_memory = SYSTEM + memory_context
    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")

    # Only use tools for deep Islamic knowledge queries
    tool_keywords = [
        'hadith', 'quran verse', 'surah', 'ayah', 'ruling', 'fiqh',
        'fatwa', 'haram ruling', 'halal ruling', 'scholar said',
        'madhab', 'daleel', 'evidence from', 'proof from', 'sunnah of'
    ]
    use_tools = any(word in last_user_msg.lower() for word in tool_keywords)

    # Build full conversation history
    full_messages = [SystemMessage(content=system_with_memory)]
    for m in messages[:-1]:
        if m.role == "user":
            full_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            full_messages.append(AIMessage(content=m.content))

    if use_tools:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: chatbot.invoke({"messages": HumanMessage(content=last_user_msg)}, config=config)
                ),
                timeout=8.0
            )
            tool_context = []
            for m in result["messages"]:
                if isinstance(m, ToolMessage):
                    tool_context.append(str(m.content))
            research = "\n\n---\n\n".join(tool_context) if tool_context else ""
        except asyncio.TimeoutError:
            print("Tool search timed out")
            research = ""
        except Exception as e:
            print(f"Tool error: {e}")
            research = ""

        final_msg = f"{last_user_msg}\n\nResearch from Islamic sources:\n{research}" if research else last_user_msg
    else:
        final_msg = last_user_msg

    full_messages.append(HumanMessage(content=final_msg))

    # Stream with fast model
    async for chunk in llm_stream.astream(full_messages):
        token = chunk.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0)

    yield f"data: {json.dumps({'done': True})}\n\n"


async def stream_tafsir_agent(surah: str, ayah: int, arabic: str, translation: str):
    from langchain_core.messages import HumanMessage, SystemMessage

    loop = asyncio.get_event_loop()

    # Load agent in thread so it doesn't block the event loop
    _, retriever, _, llm_stream = await loop.run_in_executor(None, get_agent)

    # Retrieve relevant FAISS context in thread
    query = f"tafsir {surah} ayah {ayah} {translation[:60]}"
    docs = await loop.run_in_executor(None, retriever.invoke, query)
    context = "\n".join([doc.page_content for doc in docs[:2]]) if docs else ""

    tafsir_system = (
        "You are Nur, a knowledgeable Islamic scholar AI. "
        "Give a concise, spiritually uplifting tafsir in 2-3 sentences. "
        "Mention one classical scholar (Ibn Kathir, Al-Tabari, or Al-Qurtubi) naturally. "
        "Warm, scholarly tone. No bullet points. "
        "Respond in the EXACT same language as the translation provided."
    )

    user_prompt = (
        f"Surah {surah}, Ayah {ayah}:\n"
        f"Arabic: \"{arabic}\"\nTranslation: \"{translation}\"\n"
    )
    if context:
        user_prompt += f"\nRelevant Islamic knowledge:\n{context}\n"

    messages = [SystemMessage(content=tafsir_system), HumanMessage(content=user_prompt)]

    async for chunk in llm_stream.astream(messages):
        token = chunk.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0)
    yield f"data: {json.dumps({'done': True})}\n\n"


_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

async def stream_simple(prompt: str, system: str = None, max_tokens: int = None):
    client = get_openai_client()
    kwargs = dict(
        model="gpt-4o-mini",
        temperature=0.3,
        stream=True,
        messages=[
            {"role": "system", "content": system or SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0)
    yield f"data: {json.dumps({'done': True})}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Nur backend running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages found")
    user_id = req.thread_id or 'anonymous'
    if not req.is_pro and user_id not in _SYSTEM_THREAD_IDS:
        if not check_and_increment(_chat_limits, user_id, FREE_CHAT_LIMIT):
            raise HTTPException(status_code=429, detail="Daily message limit reached. Upgrade to Pro for unlimited messages.")
    return StreamingResponse(
        stream_agent_response(
            req.messages,
            thread_id=req.thread_id,
            user_memory=req.user_memory,
            convo_summaries=req.convo_summaries,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/explain-verse")
async def explain_verse(req: VerseExplainRequest):
    if req.user_id and not req.is_pro:
        if not check_and_increment(_tafsir_limits, req.user_id, FREE_TAFSIR_LIMIT):
            raise HTTPException(status_code=429, detail="Daily tafsir limit reached. Upgrade to Pro for unlimited explanations.")
    prompt = (
        f'You are Nur, a deeply knowledgeable Islamic scholar AI. '
        f'Provide a rich, spiritually uplifting tafsir for Surah {req.surah}, Ayah {req.ayah}:\n'
        f'Arabic: "{req.arabic}"\nTranslation: "{req.translation}"\n\n'
        f'Structure your tafsir in this order:\n'
        f'1. The core meaning of the ayah — what Allah is conveying\n'
        f'2. Historical/revelation context (asbab al-nuzul) if known\n'
        f'3. Insights from AT LEAST 2 classical scholars — you MUST name them explicitly. '
        f'Choose from: Ibn Kathir, Imam Al-Tabari, Imam Al-Qurtubi, Imam Al-Baghawi, Ibn Ashur, Al-Razi, Al-Zamakhshari. '
        f'Show the diversity of scholarly thought — do not give only one perspective.\n'
        f'4. A practical, actionable lesson for daily Muslim life today\n\n'
        f'Write in flowing, connected prose (not bullet points). 5-7 sentences total. '
        f'Be warm, scholarly, and spiritually engaging.\n'
        f'CRITICAL: Respond in the EXACT same language as the translation above. '
        f'If translation is French → respond in French. If English → respond in English.'
    )
    return StreamingResponse(
        stream_simple(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/daily-lesson")
async def daily_lesson(req: LessonRequest):
    import random
    topics = [
        "The 99 Names of Allah — Al-Wadud (The Most Loving)",
        "Understanding Tawakkul — complete trust in Allah",
        "The concept of Barakah and how to invite it",
        "Ihsan — worshipping Allah as though you see Him",
        "The importance of Istighfar in daily life",
        "The levels of Sabr (patience) in Islam",
        "What is the concept of Nafs in Islamic psychology?",
        "The spiritual significance of Salah beyond obligation",
    ]

    topic = req.topic or random.choice(topics)
    lang_instruction = "Respond in French." if req.lang == 'fr' else "Respond in English."
    
    prompt = (
        f'You are Nur, a warm and knowledgeable Islamic companion. '
        f'Create an inspiring 2-minute daily Islamic lesson on: "{topic}".\n\n'
        f'Format exactly as follows:\n'
        f'- First line: a compelling title (no label, just the title)\n'
        f'- One Arabic term central to the topic with transliteration and meaning\n'
        f'- 2-3 sentences of rich explanation grounded in Quran or authentic Hadith — cite the reference\n'
        f'- One sentence naming a classical or contemporary scholar\'s insight on this topic\n'
        f'- End with one practical action the reader can take today\n\n'
        f'Be uplifting, concise, and spiritually motivating. '
        f'{lang_instruction}'
    )
    return StreamingResponse(
        stream_simple(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/reflection-prompt")
async def reflection_prompt(req: ReflectionRequest):
    prompts = {
        "fajr":    "You are Nur, a spiritual Islamic companion. Write one deep, personal reflection question for after Fajr prayer. Focus on intention, gratitude, and starting the day with Allah's remembrance. Make it thought-provoking and personal — not generic. One sentence only.",
        "dhuhr":   "You are Nur, a spiritual Islamic companion. Write one meaningful reflection question for midday. Focus on staying conscious of Allah during the busy hours, checking one's actions and intentions. One sentence only.",
        "asr":     "You are Nur, a spiritual Islamic companion. Write one honest reflection question for the afternoon. Focus on accountability — have I been my best Muslim self today? One sentence only.",
        "maghrib": "You are Nur, a spiritual Islamic companion. Write one heartfelt reflection question for sunset. Focus on gratitude for what Allah gave today and the beauty of His creation. One sentence only.",
        "isha":    "You are Nur, a spiritual Islamic companion. Write one soul-searching reflection question for the night. Focus on forgiveness, repentance, and ending the day in a state of peace with Allah. One sentence only.",
        "general": "You are Nur, a spiritual Islamic companion. Write one profound Islamic reflection question that stirs the heart and encourages genuine spiritual growth. Make it personal and deep. One sentence only.",
    }
    prompt = prompts.get(req.prayer_time, prompts["general"])
    return StreamingResponse(
        stream_simple(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/halal-finder")
async def halal_finder(lat: float, lng: float, query: str, radius: int = 5000):
    import httpx
    GOOGLE_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.currentOpeningHours,places.location,places.id",
    }
    payload = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius),
            }
        },
        "maxResultCount": 20,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        data = response.json()
    places = []
    for p in data.get("places", []):
        places.append({
            "place_id": p.get("id", ""),
            "name": p.get("displayName", {}).get("text", ""),
            "vicinity": p.get("formattedAddress", ""),
            "rating": p.get("rating"),
            "opening_hours": {"open_now": p.get("currentOpeningHours", {}).get("openNow")},
            "geometry": {"location": {
                "lat": p.get("location", {}).get("latitude", 0),
                "lng": p.get("location", {}).get("longitude", 0),
            }},
        })
    return {"results": places, "status": "OK"}

# ── Halal image analysis ───────────────────────────────────────────────────────
from pydantic import BaseModel
from typing import Optional
import base64

class HalalImageRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/api/analyze-halal-image")
async def analyze_halal_image(req: HalalImageRequest):
    try:
        # 1. Clean the base64 string (Remove metadata prefix if present)
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        prompt = """You are an expert halal food analyst. Carefully examine this ingredients label image.

        Step 1: Extract ALL visible text from the ingredients list — read every word carefully.
        Step 2: Identify the product name if visible.
        Step 3: Analyze each ingredient against Islamic halal guidelines.

        HARAM ingredients (forbidden): pork, pig, swine, lard, gelatin (unless labeled fish/plant/halal),
        alcohol, wine, beer, ethanol, blood, carmine, cochineal, E120, E441, ham, bacon, pepperoni,
        animal rennet, L-cysteine from hair (E920).

        DOUBTFUL ingredients (mashbooh): natural flavors, artificial flavors, mono and diglycerides (E471),
        E472, E473, E474, E475, whey, casein, vanilla extract, lecithin (E322), E476, gelatin (source unknown).

        Respond ONLY in this exact JSON format with no extra text:
        {
          "status": "halal" or "haram" or "doubtful",
          "productName": "product name if visible, else Unknown",
          "ingredients": "full ingredients text exactly as you read it",
          "found": ["list of specific concerning ingredients found"],
          "verdict": "clear one sentence verdict",
          "explanation": "2-3 sentences explaining the halal status with specific reasons"
        }"""

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def _call():
            return client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }],
            )

        message = await asyncio.get_event_loop().run_in_executor(None, _call)

        # 3. Robust JSON parsing
        text = message.choices[0].message.content
        # Remove markdown code blocks if the model included them
        clean = text.replace('```json', '').replace('```', '').strip()
        result = json.loads(clean)
        return result
        
    except Exception as e:
        print(f"Halal image analysis error: {str(e)}")
        # Return a graceful fallback that matches the frontend expected shape
        return {
            "status": "doubtful",
            "productName": "Analysis Failed",
            "ingredients": "",
            "found": [],
            "verdict": "Error processing image",
            "explanation": f"The AI encountered an error: {str(e)[:50]}. Please try manual entry."
        }
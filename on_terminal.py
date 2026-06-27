"""
Hierarchical Memory Chatbot 
4 layers: Buffer | Summary | Vector | Entity

"""

import re, sys, os, numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

BUFFER_LIMIT    = 5
SUMMARY_TRIGGER = 10
VECTOR_TOP_K    = 3
MODEL           = "llama-3.3-70b-versatile"
client          = Groq(api_key=os.getenv("API_KEY"))


def embed(text):
    """Hash-based 100-dim vector, L2-normalized."""
    words = re.findall(r'\w+', text.lower())
    vec   = np.zeros(100)
    for w in words:
        vec[hash(w) % 100] += 1
    n = np.linalg.norm(vec)
    return vec / n if n else vec

def llm(system, messages):
    r = client.chat.completions.create(
        model=MODEL, max_tokens=1000,
        messages=[{"role": "system", "content": system}, *messages]
    )
    return r.choices[0].message.content

def summarize(messages, existing=""):
    history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    prefix  = f"Existing summary:\n{existing}\n\n" if existing else ""
    return llm("You are a summarizer.", [{
        "role": "user",
        "content": f"{prefix}Summarize in 2-3 sentences, keep names and key points:\n\n{history}"
    }])


state = {
    "buffer":   [],   
    "summary":  "",  
    "vectors":  [],  
    "entities": {}, 
    "turn":     0,
    "history":  [],  
}

ENTITY_RE = {
    "person":  r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
    "product": r'\b(Python|LangChain|Groq|Claude|GPT|API|SDK|LLaMA|iPhone)\b',
}


def ingest(role, text):
    s = state
    s["turn"] += 1


    s["buffer"].append({"role": role, "content": text})
    if len(s["buffer"]) > BUFFER_LIMIT:
        s["buffer"].pop(0)

    
    s["history"].append({"role": role, "content": text})
    if s["turn"] % SUMMARY_TRIGGER == 0:
        older = s["history"][:-BUFFER_LIMIT]
        s["summary"] = summarize(older, s["summary"])
        print(f"\n[Layer 2] Summary updated.")

    
    s["vectors"].append({"role": role, "content": text,
                         "emb": embed(text), "turn": s["turn"]})

    
    for etype, pattern in ENTITY_RE.items():
        for name in re.findall(pattern, text):
            if name not in s["entities"]:
                s["entities"][name] = {"type": etype, "count": 0}
                print(f"[Layer 4] Entity: {name!r} ({etype})")
            s["entities"][name]["count"] += 1

def build_context(query):
    s = state
    parts = ["You are a helpful support assistant."]

    if s["summary"]:
        parts.append(f"\n## Conversation summary\n{s['summary']}")

    if s["entities"]:
        lines = "\n".join(f"- {n} ({v['type']}, {v['count']}x)"
                          for n, v in s["entities"].items())
        parts.append(f"\n## Tracked entities\n{lines}")

    q_emb = embed(query)
    hits  = sorted(
        [v for v in s["vectors"] if v["turn"] != s["turn"]],
        key=lambda v: float(np.dot(q_emb, v["emb"])), reverse=True
    )[:VECTOR_TOP_K]
    if hits:
        lines = "\n".join(f"- {h['role'].upper()}: {h['content']}" for h in hits)
        parts.append(f"\n## Relevant past messages\n{lines}")

    return "\n".join(parts), s["buffer"]

def chat():
    print("Hierarchical Memory Chatbot  |  'exit' to quit  |  'debug' for state")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input: continue
        if user_input.lower() in ("exit", "quit"): break
        if user_input.lower() == "debug":
            s = state
            print(f"Turn:{s['turn']} Buffer:{len(s['buffer'])} "
                  f"Entities:{list(s['entities'].keys())}")
            continue

        ingest("user", user_input)
        system, messages = build_context(user_input)
        reply = llm(system, messages)
        ingest("assistant", reply)
        print(f"\nAgent: {reply}")

def demo():
    print("DEMO MODE — no LLM calls\n")
    turns = [
        ("user",      "Hi, I'm John Smith working on a Python project."),
        ("assistant", "Hello John! Happy to help."),
        ("user",      "I use LangChain but have memory issues."),
        ("assistant", "Try ConversationBufferMemory."),
        ("user",      "I also use the Groq API and streaming breaks."),
        ("assistant", "Pass stream=True to chat.completions.create()."),
        ("user",      "What did John say about Groq?"),
    ]
    for role, text in turns:
        ingest(role, text)
        if role == "user":
            system, buf = build_context(text)
            print(f"Turn {state['turn']}: {text}")
            print(f"  buffer={len(buf)}  entities={list(state['entities'].keys())}")
            q_emb = embed(text)
            hits = sorted(
                [v for v in state["vectors"] if v["turn"] != state["turn"]],
                key=lambda v: float(np.dot(q_emb, v["emb"])), reverse=True
            )[:1]
            if hits: print(f"  top vector hit: {hits[0]['content'][:60]}")
            print()

if __name__ == "__main__":
    demo() if len(sys.argv) > 1 and sys.argv[1] == "demo" else chat()
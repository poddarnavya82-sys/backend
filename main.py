import asyncio
import json
import os
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(
    title="AEGIS Multi-Agent Research & Debate Engine",
    description="Asynchronous SSE streaming engine for multi-agent competitive analysis"
)

# Enable CORS so frontend (Vercel) can talk to backend (Render) smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECURE API KEY SETUP: Reads from OS Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class DebateRequest(BaseModel):
    topic: str
    agent_a_profile: str = "Techno-Optimist"
    agent_b_profile: str = "Cautious Tech-Skeptic"
    rounds: int = 3

async def run_debate_lifecycle(topic: str, agent_a: str, agent_b: str, total_rounds: int) -> AsyncGenerator[str, None]:
    """
    Asynchronous State Machine running a turn-based cognitive debate.
    Streams structured Server-Sent Events (SSE) back to the client interface.
    """
    if not GEMINI_API_KEY:
        yield "data: " + json.dumps({
            "event": "error", 
            "message": "GEMINI_API_KEY is missing on the server. Please add your key to Render environment variables."
        }) + "\n\n"
        return

    # Initialize the fast stable Gemini flash model with the correct prefix path
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
    except Exception as e:
        yield "data: " + json.dumps({
            "event": "error",
            "message": f"Failed to initialize Gemini Model: {str(e)}"
        }) + "\n\n"
        return

    debate_history = []

    # Send start event to initialize UI monitors on Vercel
    yield "data: " + json.dumps({
        "event": "start", 
        "topic": topic,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "total_rounds": total_rounds
    }) + "\n\n"
    await asyncio.sleep(0.2)

    # Begin Turn-Based Debate Loop
    for r in range(1, total_rounds + 1):
        
        # ----------------------------------------------------
        # TURN 1: AGENT ALPHA (PRO)
        # ----------------------------------------------------
        yield "data: " + json.dumps({
            "event": "status", 
            "message": f"Agent A is formulating constructive arguments for Round {r}..."
        }) + "\n\n"
        await asyncio.sleep(0.1)

        history_str = "\n\n".join([
            f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}): {item['text']}" 
            for item in debate_history
        ])

        prompt_a = (
            f"You are Agent A, arguing IN FAVOR of: '{topic}'.\n"
            f"Your specific persona is: {agent_a}.\n"
            f"Strictly adopt this character. Address and systematically dismantle any previous counterpoints.\n"
            f"Keep your response razor-sharp, highly logical, professional, and under 150 words. This is Round {r} of {total_rounds}.\n\n"
            f"--- DEBATE HISTORY SO FAR ---\n"
            f"{history_str if history_str else '[This is your opening statement. Establish your thesis clearly.]'}\n\n"
            f"Now, deliver your argument:"
        )

        yield "data: " + json.dumps({"event": "stream_start", "agent": "A", "round": r}) + "\n\n"

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt_a,
                stream=True
            )
            
            accumulated_a_text = ""
            for chunk in response:
                try:
                    if chunk.text:
                        accumulated_a_text += chunk.text
                        yield "data: " + json.dumps({
                            "event": "stream_chunk", 
                            "agent": "A", 
                            "text": chunk.text
                        }) + "\n\n"
                        await asyncio.sleep(0.005) # Elegant pacing for streaming
                except Exception:
                    continue

            debate_history.append({"round": r, "agent": "A", "text": accumulated_a_text})
            
            yield "data: " + json.dumps({
                "event": "stream_end", 
                "agent": "A", 
                "round": r, 
                "full_text": accumulated_a_text
            }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Agent A generation failed: {str(e)}"}) + "\n\n"
            return

        await asyncio.sleep(0.5)

        # ----------------------------------------------------
        # TURN 2: AGENT BETA (CON)
        # ----------------------------------------------------
        yield "data: " + json.dumps({
            "event": "status", 
            "message": f"Agent B is scanning arguments and formulating Round {r} counter-rebuttal..."
        }) + "\n\n"
        await asyncio.sleep(0.1)

        history_str = "\n\n".join([
            f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}): {item['text']}" 
            for item in debate_history
        ])

        prompt_b = (
            f"You are Agent B, arguing critically AGAINST the topic: '{topic}'.\n"
            f"Your specific persona is: {agent_b}.\n"
            f"Directly analyze, attack, and exploit logical vulnerabilities in Agent A's arguments.\n"
            f"Keep your response brilliant, persuasive, critical, and under 150 words. This is Round {r} of {total_rounds}.\n\n"
            f"--- DEBATE HISTORY SO FAR ---\n"
            f"{history_str}\n\n"
            f"Now, deliver your counterpoint:"
        )

        yield "data: " + json.dumps({"event": "stream_start", "agent": "B", "round": r}) + "\n\n"

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt_b,
                stream=True
            )
            
            accumulated_b_text = ""
            for chunk in response:
                try:
                    if chunk.text:
                        accumulated_b_text += chunk.text
                        yield "data: " + json.dumps({
                            "event": "stream_chunk", 
                            "agent": "B", 
                            "text": chunk.text
                        }) + "\n\n"
                        await asyncio.sleep(0.005)
                except Exception:
                    continue

            debate_history.append({"round": r, "agent": "B", "text": accumulated_b_text})
            
            yield "data: " + json.dumps({
                "event": "stream_end", 
                "agent": "B", 
                "round": r, 
                "full_text": accumulated_b_text
            }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Agent B generation failed: {str(e)}"}) + "\n\n"
            return

        await asyncio.sleep(0.5)

    # ----------------------------------------------------
    # FINAL PHASE: THE SUPREME AI JUDGE
    # ----------------------------------------------------
    yield "data: " + json.dumps({
        "event": "status", 
        "message": "Debate concluded. Generating holistic Judicial Metric Analysis..."
    }) + "\n\n"
    await asyncio.sleep(0.5)

    transcript_summary = "\n\n".join([
        f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}):\n{item['text']}" 
        for item in debate_history
    ])

    judge_prompt = (
        f"You are the Supreme Court AI Judge, an objective, hyper-rational adjudicator.\n"
        f"Analyze the following {total_rounds}-round transcript of a debate on: '{topic}'.\n\n"
        f"--- FULL DEBATE TRANSCRIPT ---\n"
        f"{transcript_summary}\n\n"
        f"Provide a definitive and deeply reasoned verdict formatted in elegant, structured Markdown.\n"
        f"You MUST format the output to contain these precise sections:\n"
        f"1. # Executive Verdict (Explicitly state the winner, a quantitative score like 88-84, and the core reasoning)\n"
        f"2. ## Agent A Arguments appraisal (Highlighting strengths and missed opportunities)\n"
        f"3. ## Agent B Arguments appraisal (Highlighting strengths and missed opportunities)\n"
        f"4. ## Rebuttal Clashes Analysis (Analyse how well they actually listened and counter-attacked)\n"
        f"5. ## Scoring Breakdown (A Markdown Table covering: Logical Rigor, Rhetoric, Persuasiveness, Rebuttals)\n"
        f"6. ## Core Post-Debate Synthesis (What key underlying truth does this clash reveal?)"
    )

    yield "data: " + json.dumps({"event": "judge_start"}) + "\n\n"

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            judge_prompt,
            stream=True
        )

        for chunk in response:
            try:
                if chunk.text:
                    yield "data: " + json.dumps({
                        "event": "judge_chunk", 
                        "text": chunk.text
                    }) + "\n\n"
                    await asyncio.sleep(0.002)
            except Exception:
                continue

    except Exception as e:
        yield "data: " + json.dumps({"event": "error", "message": f"Judge assessment failed: {str(e)}"}) + "\n\n"
        return

    # Terminate connection cleanly
    yield "data: " + json.dumps({"event": "complete"}) + "\n\n"


@app.post("/api/debate/stream")
async def debate_stream_endpoint(request: DebateRequest):
    return StreamingResponse(
        run_debate_lifecycle(
            topic=request.topic,
            agent_a=request.agent_a_profile,
            agent_b=request.agent_b_profile,
            total_rounds=request.rounds
        ),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

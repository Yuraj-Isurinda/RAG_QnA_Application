import React, { useEffect, useMemo, useRef, useState } from "react";

type ChatMessage = { role: "user" | "assistant"; content: string };

const API_BASE = import.meta.env.VITE_API_BASE || ""; // e.g. http://localhost:8000

async function askBackend(
  question: string,
  history: string[],
  signal?: AbortSignal,
): Promise<{ answer: string }> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const retryAfter = res.headers.get("Retry-After");
    const msg = `HTTP ${res.status} ${res.statusText}\n${text}`;
    const err: any = new Error(msg);
    if (retryAfter) err.retryAfter = parseInt(retryAfter, 10);
    throw err;
  }
  return res.json();
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const history = useMemo(() => messages.map(m => m.content), [messages]);

  useEffect(() => {
    // Focus input on load
    (document.getElementById("user-input") as HTMLInputElement | null)?.focus();
  }, []);

  function stopRequest() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }

  async function handleSend(e?: React.FormEvent) {
    e?.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setError(null);
    setLoading(true);
    setMessages(prev => [...prev, { role: "user", content: question }]);
    setInput("");

    abortRef.current = new AbortController();
    try {
      const { answer } = await askBackend(question, history, abortRef.current.signal);
      setMessages(prev => [...prev, { role: "assistant", content: answer }]);
    } catch (err: any) {
      // Show concise errors; handle 429 backoff hints
      const is429 = /HTTP\s+429/.test(err?.message || "");
      const retryAfter = err?.retryAfter ?? undefined;
      setError(
        is429
          ? `Rate limit hit${retryAfter ? `; retry after ~${retryAfter}s` : ""}. Try again shortly or reduce context.`
          : err?.message || "Something went wrong"
      );
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="px-6 py-4 border-b bg-white">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">RAG Q&A (Gemini + Chroma)</h1>
          <div className="text-xs text-gray-500">API: {API_BASE || "/api"}</div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="rounded-2xl bg-white shadow-sm border p-4 sm:p-6 flex flex-col h-[75vh]">
          <div className="flex-1 overflow-y-auto pr-2 space-y-4" id="chat-scroll">
            {messages.length === 0 && (
              <div className="text-gray-500 text-sm">Ask anything from your PDF. I use your Python RAG backend to answer.</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`${
                    m.role === "user" ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
                  } max-w-[80%] rounded-2xl px-4 py-2 text-sm whitespace-pre-wrap`}
                >
                  {m.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-900 max-w-[80%] rounded-2xl px-4 py-2 text-sm">
                  Thinking…
                </div>
              </div>
            )}
          </div>

          <form onSubmit={handleSend} className="mt-4 flex items-end gap-2">
            <textarea
              id="user-input"
              className="flex-1 resize-none rounded-2xl border px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Type your question…"
              value={input}
              onChange={e => setInput(e.target.value)}
              rows={2}
            />
            <div className="flex flex-col gap-2">
              <button
                type="submit"
                disabled={loading}
                className="rounded-2xl px-4 py-3 text-sm bg-blue-600 text-white disabled:opacity-50"
                title="Send"
              >
                Send
              </button>
              <button
                type="button"
                onClick={stopRequest}
                disabled={!loading}
                className="rounded-2xl px-4 py-2 text-xs border disabled:opacity-50"
                title="Cancel in-flight request"
              >
                Stop
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-3 text-xs text-red-600">{error}</div>
          )}
        </div>
      </main>

      <footer className="px-6 py-4 text-center text-xs text-gray-500">Built with React + Tailwind. Backend: FastAPI (Gemini 1.5 + Chroma).</footer>
    </div>
  );
}

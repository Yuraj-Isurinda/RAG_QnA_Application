import React, { useEffect, useMemo, useRef, useState } from "react";

type ChatMessage = { role: "user" | "assistant"; content: string };

const API_BASE = import.meta.env.VITE_API_BASE || "";

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

type DocItem = {
  doc_id: string;
  display_name: string;
  path: string;
  num_chunks: number;
};

async function getDocumentsDetail(): Promise<DocItem[]> {
  const res = await fetch(`${API_BASE}/documents/detail`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  const data = await res.json();
  return data.documents as DocItem[];
}

async function deleteDocument(docId: string): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function uploadDocument(file: File): Promise<{ success: boolean; message: string; filename: string; doc_id: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [uploading, setUploading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const history = useMemo(() => messages.map((m) => m.content), [messages]);

  useEffect(() => {
    (document.getElementById("user-input") as HTMLTextAreaElement | null)?.focus();
    loadDocuments();
  }, []);

  async function loadDocuments() {
    try {
      const docs = await getDocumentsDetail();
      setDocuments(docs);
    } catch (e) {
      console.error("Failed to load documents:", e);
    }
  }

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
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");

    abortRef.current = new AbortController();
    try {
      const { answer } = await askBackend(question, history, abortRef.current.signal);
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (err: any) {
      const is429 = /HTTP\s+429/.test(err?.message || "");
      const retryAfter = err?.retryAfter ?? undefined;
      setError(
        is429
          ? `Rate limit hit${retryAfter ? `; retry after ~${retryAfter}s` : ""}. Try again shortly.`
          : err?.message || "Something went wrong"
      );
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  async function onFileChosen(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are allowed");
      e.currentTarget.value = "";
      return;
    }
    setUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await loadDocuments();
    } catch (err: any) {
      setError(err?.message || "Upload failed");
    } finally {
      setUploading(false);
      e.currentTarget.value = "";
    }
  }

  async function onDeleteDoc(d: DocItem) {
    if (!confirm(`Delete "${d.display_name}"?`)) return;
    try {
      await deleteDocument(d.doc_id);
      await loadDocuments();
    } catch (err: any) {
      setError(err?.message || "Delete failed");
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="px-6 py-4 border-b bg-white">
        <div className="max-w-4xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-semibold tracking-tight">ContextLens</h1>
          </div>

          <div className="flex items-center gap-3">

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={onFileChosen}
              className="hidden"
            />
            <button
              onClick={openFilePicker}
              disabled={uploading}
              className="rounded-2xl inline-flex items-center gap-2 bg-blue-600 text-white px-6 py-2 text-sm disabled:opacity-50"
              title="Upload PDF"
            >
              {uploading ? "Uploading…" : "Upload PDF"}
            </button>

            
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="rounded-2xl bg-white shadow-sm border p-4 sm:p-6 flex flex-col h-[75vh]">

          {/* Documents panel */}
          {documents.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Documents</h3>
              <ul className="text-sm text-gray-700 divide-y rounded-lg border">
                {documents.map((d) => (
                  <li key={d.doc_id} className="flex items-center justify-between px-3 py-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="truncate" title={d.path}>{d.display_name}</span>
                      {d.num_chunks === 0 && (
                        <span
                          title="No text extracted; this may be a scanned PDF. Consider OCR."
                          className="inline-flex items-center rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 text-[11px]"
                        >
                          OCR suggested
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => onDeleteDoc(d)}
                      className="p-1.5 rounded hover:bg-red-50"
                      aria-label={`Delete ${d.display_name}`}
                      title="Delete"
                    >
                      {/* Trash icon */}
                      <img src="/trash.svg" alt="Delete" className="object-contain" style={{ width: "30px", height: "30px" }} />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Chat messages */}
          <div className="flex-1 overflow-y-auto pr-2 space-y-4" id="chat-scroll">
            {messages.length === 0 && (
              <div className="text-gray-500 text-sm">
                Upload a PDF, then ask questions.
              </div>
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

          {/* Composer */}
          <form onSubmit={handleSend} className="mt-4 flex items-end gap-2">
            <textarea
              id="user-input"
              className="flex-1 resize-none rounded-2xl border px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder={documents.length === 0 ? "Upload a PDF to start…" : "Type your question…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={2}
              disabled={documents.length === 0 || loading}
            />
            <div className="flex flex-col gap-2">
              <button
                type="submit"
                disabled={loading || documents.length === 0}
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
          <span className="text-s text-gray-500">Docs: {documents.length}</span>
          {error && <div className="mt-3 text-xs text-red-600">{error}</div>}
        </div>
      </main>
      <footer className="px-6 py-4 text-center text-xs text-gray-500">
        Made with love by Yuraj Isurinda
      </footer>
    </div>
  );
}

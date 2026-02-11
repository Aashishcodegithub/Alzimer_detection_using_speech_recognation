import React, { useState, useEffect, useRef } from "react";
import {
  Mic,
  Upload,
  Activity,
  Brain,
  User,
  LogOut,
  ChevronRight,
  Lock,
  Mail,
  UserPlus,
  FileAudio,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  MessageSquare,
  X,
  Send,
  Loader2,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";
const GEMINI_API_KEY = "AIzaSyAxVaesEB2skVSneYdU17zNFuQNVliRwk8";

const ConfidenceBar = ({ probability }) => {

  const raw = typeof probability === "number" ? probability : 0.5;
  const p = Math.min(0.99, Math.max(0.01, raw));
  const pct = Math.round(p * 100);

  let label = "Medium";
  let colorClass =
    "bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600";

  if (p < 0.33) {
    label = "Low";
    colorClass = "bg-gradient-to-r from-emerald-400 to-emerald-500";
  } else if (p >= 0.66) {
    label = "High";
    colorClass = "bg-gradient-to-r from-rose-400 to-red-500";
  }

  return (
    <div className="mt-4">
      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
        <span>Confidence band</span>
        <span className="font-semibold text-slate-200">{label}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-800/80 overflow-hidden">
        <div
          className={`h-full ${colorClass} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500 mt-1">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
};


const generateGeminiContent = async (prompt, systemInstruction = "") => {
  const delays = [1000, 2000, 4000, 8000, 16000];
  let attempt = 0;

  while (attempt <= 5) {
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            systemInstruction: systemInstruction
              ? { parts: [{ text: systemInstruction }] }
              : undefined,
          }),
        }
      );

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("Rate limit");
        }
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error?.message || "Gemini API Error");
      }

      const data = await response.json();
      return (
        data.candidates?.[0]?.content?.parts?.[0]?.text ||
        "No response generated."
      );
    } catch (err) {
      if (attempt === 5) throw err;
      await new Promise((resolve) => setTimeout(resolve, delays[attempt]));
      attempt++;
    }
  }
};

const CustomStyles = () => (
  <style>{`
    @keyframes blob {
      0% { transform: translate(0px, 0px) scale(1); }
      33% { transform: translate(30px, -50px) scale(1.1); }
      66% { transform: translate(-20px, 20px) scale(0.9); }
      100% { transform: translate(0px, 0px) scale(1); }
    }
    @keyframes float {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-20px); }
      100% { transform: translateY(0px); }
    }
    @keyframes pulse-ring {
      0% { transform: scale(0.8); opacity: 0.5; }
      100% { transform: scale(2); opacity: 0; }
    }
    @keyframes spin-slow {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    @keyframes scan {
      0% { top: 0%; opacity: 0; }
      10% { opacity: 1; }
      90% { opacity: 1; }
      100% { top: 100%; opacity: 0; }
    }
    @keyframes slide-up {
      from { transform: translateY(100%); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    .animate-blob { animation: blob 10s infinite; }
    .animate-float { animation: float 6s ease-in-out infinite; }
    .animate-spin-slow { animation: spin-slow 12s linear infinite; }
    .animate-scan { animation: scan 3s linear infinite; }
    .animate-slide-up { animation: slide-up 0.3s ease-out forwards; }

    .glass-panel {
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-input {
      background: rgba(30, 41, 59, 0.5);
      backdrop-filter: blur(4px);
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .glass-input:focus {
      background: rgba(30, 41, 59, 0.8);
      border-color: rgba(139, 92, 246, 0.5);
      box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1);
    }
    .scrollbar-hide::-webkit-scrollbar {
      display: none;
    }
    .typing-dot {
      animation: typing 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1); }
    }
  `}</style>
);

export default function App() {
  const [currentView, setCurrentView] = useState("login");

  const [user, setUser] = useState({ name: "", email: "", token: "" });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [audioFile, setAudioFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [selectedModel, setSelectedModel] = useState("cnn_lstm"); 
  const [useEnsemble, setUseEnsemble] = useState(false);

  const [aiReport, setAiReport] = useState("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: "system",
      text: "Hello! I'm your AI Caregiver Assistant. Ask me about Alzheimer's signs, brain health tips, or how to interpret your results.",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isChatThinking, setIsChatThinking] = useState(false);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    const storedToken = sessionStorage.getItem("token");
    const storedName = sessionStorage.getItem("name");
    const storedEmail = sessionStorage.getItem("email");

    if (storedToken) {
      setUser({
        token: storedToken,
        name: storedName || "User",
        email: storedEmail || "",
      });
      setCurrentView("home");
    }
  }, []);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, showChat]);

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError("");
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
      setPrediction(null);
      setAiReport("");
      setError("");
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    if (!formData.name || !formData.email || !formData.password) {
      setError("All fields are required.");
      setIsLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Registration failed. Please try again.");
      }

      setCurrentView("login");
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    if (!formData.email || !formData.password) {
      setError("Please enter email and password.");
      setIsLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Invalid credentials.");
      }

      const data = await res.json();
      const userName = data.name || "User";
      const userData = {
        token: data.access_token,
        name: userName,
        email: formData.email,
      };

      setUser(userData);
      sessionStorage.setItem("token", userData.token);
      sessionStorage.setItem("name", userData.name);
      sessionStorage.setItem("email", userData.email);

      setCurrentView("home");
      setFormData({ name: "", email: "", password: "" });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    sessionStorage.clear();
    setUser({ name: "", email: "", token: "" });
    setCurrentView("login");
    setPrediction(null);
    setAudioFile(null);
    setAiReport("");
  };

  const handleAnalyze = async () => {
    if (!audioFile) return;
    setIsAnalyzing(true);
    setError("");
    setAiReport("");
    setPrediction(null);

    const data = new FormData();
    data.append("file", audioFile);

    data.append("model_id", selectedModel);             
    data.append("use_ensemble", useEnsemble ? "true" : "false");

    try {
      const res = await fetch(
        `${API_BASE}/predict`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${user.token}`,
          },
          body: data,
        }
      );
      if (res.status === 401) {
        handleLogout();
        throw new Error("Session expired. Please login again.");
      }

      if (!res.ok) {
        throw new Error("Analysis failed. Please try a different file.");
      }

      const result = await res.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message || "Something went wrong during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };


const handleGenerateReport = async () => {
 
  if (!prediction || prediction.probability === undefined) {
    setError("Please run diagnosis before generating a report.");
    return;
  }

  setLoadingReport(true);
  setAiReport("");
  setError("");

  const safeModelName =
    prediction.model_name ||
    (useEnsemble ? "Ensemble" : "CNN-LSTM");

  try {
    const res = await fetch("http://127.0.0.1:8000/gemini/alz-chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        question:
          "Explain this Alzheimer's screening result for a caregiver. " +
          "Include the confidence level, what the result suggests, and recommended next steps.",

        class_name: prediction.class_name,
        probability: prediction.probability,
        model_name: safeModelName,
      }),
    });

    if (!res.ok) {
      throw new Error("AI report service unavailable");
    }

    const data = await res.json();


    setAiReport(
      data.answer ||
        "AI report could not be generated at this time. " +
          "Please review the confidence score and consult a healthcare professional."
    );
  } catch (err) {
    setAiReport(
      "AI report generation is temporarily unavailable. " +
        "This screening result is based on voice analysis and should be verified by a neurologist."
    );
  } finally {
    setLoadingReport(false);
  }
};

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { role: "user", text: chatInput };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    setIsChatThinking(true);

    const systemPrompt =
      "You are a compassionate, knowledgeable AI assistant for an Alzheimer's Voice Screening application. Your goal is to provide supportive, scientifically accurate (but accessible) information about brain health, Alzheimer's early signs, and how voice analysis technology generally works. Always remind users you are an AI, not a doctor. Be concise.";

    try {
      const conversation = chatMessages
        .slice(-4)
        .map(
          (m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`
        )
        .join("\n");
      const prompt = `${conversation}\nUser: ${chatInput}\nAssistant:`;

      const response = await generateGeminiContent(prompt, systemPrompt);
      setChatMessages((prev) => [...prev, { role: "ai", text: response }]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "I'm having trouble connecting right now. Please try again.",
        },
      ]);
    } finally {
      setIsChatThinking(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#020617] text-slate-200 font-sans selection:bg-violet-500/30 relative overflow-hidden">
      <CustomStyles />

      {/* global background blobs/noise */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-violet-600/20 rounded-full blur-[120px] animate-blob" />
        <div
          className="absolute top-[20%] right-[-10%] w-[400px] h-[400px] bg-indigo-600/20 rounded-full blur-[100px] animate-blob"
          style={{ animationDelay: "2s" }}
        />
        <div
          className="absolute bottom-[-10%] left-[20%] w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[120px] animate-blob"
          style={{ animationDelay: "4s" }}
        />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150 mix-blend-overlay" />
      </div>

      {/* AUTH VIEWS */}
      {(currentView === "login" || currentView === "register") && (
        <div className="relative z-10 min-h-screen flex items-center justify-center px-4 py-10">
          {/* auth-area blobs */}
          <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
            <div className="absolute -top-32 -left-24 h-72 w-72 rounded-full bg-violet-600/30 blur-3xl animate-blob" />
            <div
              className="absolute -bottom-32 -right-24 h-80 w-80 rounded-full bg-indigo-600/25 blur-3xl animate-blob"
              style={{ animationDelay: "2s" }}
            />
            <div className="absolute top-1/2 left-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full border border-violet-500/20 bg-violet-500/5 blur-3xl" />
          </div>

          <div className="w-full max-w-6xl grid gap-10 lg:grid-cols-2 items-center">
            {/* LEFT HERO PANEL */}
            <div className="space-y-8 text-center lg:text-left">
              <div className="inline-flex items-center gap-3 rounded-full border border-violet-500/30 bg-violet-500/10 px-3 py-1 text-xs font-medium text-violet-200 shadow-lg shadow-violet-500/20 backdrop-blur">
                <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-violet-500/80">
                  🧠
                </span>
                AI-Powered Alzheimer’s Voice Screening
              </div>

              <div className="space-y-4">
                <h1 className="text-4xl md:text-5xl font-extrabold leading-tight text-transparent bg-clip-text bg-gradient-to-br from-white via-violet-200 to-slate-400">
                  Early detection,
                  <br className="hidden md:block" /> gentle experience.
                </h1>
                <p className="text-slate-400 text-sm md:text-base max-w-xl mx-auto lg:mx-0">
                  Upload short voice samples and let our  model assist
                  you in screening early signs of Alzheimer’s disease. Designed
                  for doctors, caregivers, and research labs.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-3 text-xs md:text-sm">
                <div className="glass-panel rounded-2xl p-4 flex flex-col gap-2">
                  <span className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
                    Accuracy
                  </span>
                  <span className="text-lg font-semibold text-violet-200">
                    90%+
                  </span>
                  <span className="text-slate-500">CNN-LSTM baseline</span>
                </div>
                <div className="glass-panel rounded-2xl p-4 flex flex-col gap-2">
                  <span className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
                    Input
                  </span>
                  <span className="text-lg font-semibold text-violet-200">
                    5–10 sec
                  </span>
                  <span className="text-slate-500">Natural speech</span>
                </div>
                <div className="glass-panel rounded-2xl p-4 flex flex-col gap-2">
                  <span className="text-[10px] uppercase tracking-[0.18em] text-slate-400">
                    Mode
                  </span>
                  <span className="text-lg font-semibold text-violet-200">
                    Clinical
                  </span>
                  <span className="text-slate-500">Non-diagnostic tool</span>
                </div>
              </div>

              <div className="flex items-center justify-center lg:justify-start gap-3 text-xs text-slate-500">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span>Secure &amp; encrypted</span>
                </div>
                <span className="hidden md:inline text-slate-600">•</span>
                <span className="hidden md:inline">
                  Built for research &amp; clinical pilots
                </span>
              </div>
            </div>

            {/* RIGHT AUTH CARD */}
            <div className="relative">
              <div className="absolute inset-0 -z-10 scale-110 rounded-[32px] bg-gradient-to-br from-violet-500/50 via-transparent to-indigo-500/40 blur-2xl opacity-60" />

              <div className="glass-panel rounded-[28px] p-8 md:p-10 shadow-2xl shadow-black/40 border border-white/10">
                <div className="flex items-center justify-center mb-6">
                  <div className="relative">
                    <div className="absolute inset-0 rounded-2xl bg-violet-500/30 blur-lg animate-pulse" />
                    <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-500 shadow-lg shadow-violet-500/40 animate-float">
                      <Brain className="h-6 w-6 text-white" />
                    </div>
                  </div>
                </div>

                <div className="text-center mb-6 space-y-1">
                  <h2 className="text-2xl font-bold text-slate-50">
                    {currentView === "login"
                      ? "Welcome back"
                      : "Create an account"}
                  </h2>
                  <p className="text-xs text-slate-400">
                    {currentView === "login"
                      ? "Sign in to access voice analysis and AI reports."
                      : "Start screening voice samples in a secure dashboard."}
                  </p>
                </div>

                <form
                  onSubmit={currentView === "login" ? handleLogin : handleRegister}
                  className="space-y-5"
                >
                  {currentView === "register" && (
                    <div className="space-y-2">
                      <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.18em] ml-1">
                        Full name
                      </label>
                      <div className="relative group">
                        <User className="absolute left-4 top-3.5 h-5 w-5 text-slate-500 group-focus-within:text-violet-400 transition-colors" />
                        <input
                          name="name"
                          type="text"
                          placeholder="Abhinav Singh"
                          value={formData.name}
                          onChange={handleInputChange}
                          className="w-full glass-input rounded-2xl py-3.5 pl-12 pr-4 text-sm text-white placeholder-slate-500 outline-none transition-all"
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.18em] ml-1">
                      Email address
                    </label>
                    <div className="relative group">
                      <Mail className="absolute left-4 top-3.5 h-5 w-5 text-slate-500 group-focus-within:text-violet-400 transition-colors" />
                      <input
                        name="email"
                        type="email"
                        placeholder="doctor@example.com"
                        value={formData.email}
                        onChange={handleInputChange}
                        className="w-full glass-input rounded-2xl py-3.5 pl-12 pr-4 text-sm text-white placeholder-slate-500 outline-none transition-all"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-[0.18em] ml-1">
                      Password
                    </label>
                    <div className="relative group">
                      <Lock className="absolute left-4 top-3.5 h-5 w-5 text-slate-500 group-focus-within:text-violet-400 transition-colors" />
                      <input
                        name="password"
                        type="password"
                        placeholder="••••••••"
                        value={formData.password}
                        onChange={handleInputChange}
                        className="w-full glass-input rounded-2xl py-3.5 pl-12 pr-4 text-sm text-white placeholder-slate-500 outline-none transition-all"
                      />
                    </div>
                  </div>

                  {error && (
                    <div className="flex items-center gap-2 rounded-2xl border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-100">
                      <AlertCircle className="h-4 w-4" />
                      <span>{error}</span>
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={isLoading}
                    className="group relative mt-2 flex w-full items-center justify-center gap-2 overflow-hidden rounded-2xl bg-gradient-to-r from-violet-500 to-indigo-500 px-4 py-3.5 text-sm font-semibold text-white shadow-xl shadow-violet-900/40 transition-all disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/40 to-transparent opacity-0 transition-all duration-500 group-hover:translate-x-full group-hover:opacity-100" />
                    {isLoading ? (
                      <>
                        <span className="h-4 w-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        {currentView === "login"
                          ? "Sign in"
                          : "Create account"}
                        <ChevronRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
                      </>
                    )}
                  </button>
                </form>

                <div className="mt-6 flex items-center justify-between text-[11px] text-slate-400">
                  <span>
                    {currentView === "login"
                      ? "Don’t have an account?"
                      : "Already registered?"}
                  </span>
                  <button
                    onClick={() => {
                      setCurrentView(
                        currentView === "login" ? "register" : "login"
                      );
                      setError("");
                    }}
                    className="inline-flex items-center gap-1 rounded-full px-3 py-1 text-[11px] font-semibold text-violet-300 hover:text-violet-100 hover:bg-violet-500/10 transition-colors"
                  >
                    {currentView === "login" ? (
                      <>
                        <UserPlus className="h-3 w-3" />
                        Create account
                      </>
                    ) : (
                      <>
                        <LogOut className="h-3 w-3 rotate-180" />
                        Back to login
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HOME / MAIN APP VIEW */}
      {currentView === "home" && (
        <div className="relative z-10 flex flex-col h-screen">
          <header className="h-20 px-6 md:px-10 flex items-center justify-between border-b border-white/5 bg-[#020617]/80 backdrop-blur-md sticky top-0 z-50">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-slate-100 leading-tight">
                  Alzheimer Voice Lab
                </h1>
                <p className="text-[10px] uppercase tracking-widest text-violet-400 font-semibold">
                  Medical AI Assistant
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6">
              <div className="hidden md:flex flex-col items-end">
                <span className="text-sm font-medium text-slate-200">
                  Hello, {user.name}
                </span>
                <span className="text-xs text-slate-500">{user.email}</span>
              </div>
              <div className="h-8 w-[1px] bg-white/10 hidden md:block" />
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-red-500/30 hover:text-red-400 transition-all text-sm font-medium text-slate-400"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>
          </header>

          <main className="flex-1 overflow-y-auto p-4 md:p-10 scrollbar-hide">
            <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* LEFT COLUMN */}
              <div className="lg:col-span-7 space-y-6">
                <div className="glass-panel rounded-3xl p-8 relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-violet-600/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />

                  <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                    <Mic className="w-6 h-6 text-violet-400" />
                    Voice Analysis
                  </h2>
                  <p className="text-slate-400 mb-8">
                    Upload a 5–10 second .wav audio file of the patient
                    speaking naturally.
                  </p>

                  <div className="relative">
                    <input
                      type="file"
                      accept=".wav"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 z-20 cursor-pointer"
                    />
                    <div
                      className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 ${
                        audioFile
                          ? "border-violet-500 bg-violet-500/5"
                          : "border-slate-700 bg-slate-800/30 hover:border-slate-500"
                      }`}
                    >
                      {audioFile ? (
                        <div className="flex flex-col items-center">
                          <div className="w-16 h-16 rounded-2xl bg-violet-500/20 text-violet-400 flex items-center justify-center mb-4">
                            <FileAudio className="w-8 h-8" />
                          </div>
                          <h3 className="text-lg font-semibold text-white mb-1">
                            {audioFile.name}
                          </h3>
                          <p className="text-sm text-slate-400">
                            Ready for analysis
                          </p>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center group-hover:scale-105 transition-transform">
                          <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4 group-hover:bg-slate-700 transition-colors">
                            <Upload className="w-6 h-6 text-slate-400" />
                          </div>
                          <h3 className="text-lg font-semibold text-slate-200 mb-1">
                            Click to Upload Audio
                          </h3>
                          <p className="text-sm text-slate-500">
                            Supported format: .WAV (Mono, 16kHz)
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                                    {/* 🔹 Model selection & ensemble (added) */}
                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400 mb-2">
                        Model
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {[
                          { id: "cnn_lstm", label: "CNN-LSTM" },
                          { id: "gru_attn", label: "GRU-Attention" },
                        ].map((m) => (
                          <button
                            key={m.id}
                            type="button"
                            onClick={() => setSelectedModel(m.id)}
                            className={`px-3 py-1.5 rounded-full text-[11px] font-medium border transition-all ${
                              selectedModel === m.id
                                ? "bg-violet-600 text-white border-violet-500 shadow-md shadow-violet-500/30"
                                : "bg-slate-900/60 text-slate-300 border-slate-700 hover:border-slate-500"
                            }`}
                          >
                            {m.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <label className="flex items-start gap-2 cursor-pointer mt-2 md:mt-0">
                      <input
                        type="checkbox"
                        checked={useEnsemble}
                        onChange={(e) => setUseEnsemble(e.target.checked)}
                        className="mt-0.5 h-4 w-4 rounded border-slate-600 bg-slate-900"
                      />
                      <div>
                        <div className="text-xs font-semibold text-slate-200">
                          Use Ensemble
                        </div>
                        <div className="text-[11px] text-slate-500">
                          Combine all models for more stable predictions.
                        </div>
                      </div>
                    </label>
                  </div>

                  <div className="mt-6">
                    <button
                      onClick={handleAnalyze}
                      disabled={!audioFile || isAnalyzing}
                      className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-bold shadow-lg shadow-violet-900/20 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-violet-500/30 hover:-translate-y-0.5 transition-all flex items-center justify-center gap-3"
                    >
                      {isAnalyzing ? (
                        <>
                          <Activity className="w-5 h-5 animate-spin" />
                          Processing Audio...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-5 h-5" />
                          Start Diagnosis
                        </>
                      )}
                    </button>
                    {error && (
                      <p className="mt-4 text-center text-red-400 text-sm bg-red-400/10 py-2 rounded-lg border border-red-400/20">
                        {error}
                      </p>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  {[
                    { icon: Mic, label: "Record", desc: "Clear speech input" },
                    {
                      icon: Activity,
                      label: "Analyze",
                      desc: "",
                    },
                    {
                      icon: Brain,
                      label: "Predict",
                      desc: "",
                    },
                  ].map((step, idx) => {
                    const Icon = step.icon;
                    return (
                      <div
                        key={idx}
                        className="glass-panel p-4 rounded-2xl text-center"
                      >
                        <Icon className="w-6 h-6 text-slate-400 mx-auto mb-2" />
                        <div className="text-sm font-semibold text-slate-200">
                          {step.label}
                        </div>
                        <div className="text-xs text-slate-500">
                          {step.desc}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* RIGHT COLUMN */}
              <div className="lg:col-span-5">
                <div className="h-full glass-panel rounded-3xl p-8 flex flex-col relative overflow-hidden min-h-[600px]">
                  <h2 className="text-xl font-bold text-white mb-6">
                    Diagnostic Results
                  </h2>

                 {prediction ? (
                    <div className="flex-1 flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-700">
                      <div className="relative w-48 h-48 mx-auto mb-8 flex items-center justify-center">
                        <div className="absolute inset-0 border-4 border-slate-700/30 rounded-full" />
                        <div className="absolute inset-0 border-4 border-violet-500 rounded-full border-t-transparent animate-spin-slow" />
                        <div
                          className="absolute inset-4 border-2 border-indigo-400/30 rounded-full border-b-transparent animate-spin"
                          style={{ animationDuration: "8s" }}
                        />

                        <div className="text-center z-10">
                          {/* confidence number clipped 1–99% */}
                          <span className="text-4xl font-bold text-white tracking-tighter block">
                            {(
                              Math.min(0.99, Math.max(0.01, prediction.probability)) * 100
                            ).toFixed(1)}
                            %
                          </span>
                          <span className="text-xs text-violet-400 uppercase tracking-widest font-semibold">
                            Confidence
                          </span>
                        </div>
                      </div>

                      {/* 🔥 Confidence Graph Bar Added Here */}
                      <ConfidenceBar probability={prediction.probability} />

                      <div
                        className={`mb-6 p-4 rounded-2xl border ${
                          prediction.label === 1
                            ? "bg-red-500/10 border-red-500/50"
                            : "bg-emerald-500/10 border-emerald-500/50"
                        } flex items-center gap-4`}
                      >
                        <div
                          className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                            prediction.label === 1
                              ? "bg-red-500 text-white"
                              : "bg-emerald-500 text-white"
                          }`}
                        >
                          {prediction.label === 1 ? (
                            <Activity className="w-6 h-6" />
                          ) : (
                            <CheckCircle2 className="w-6 h-6" />
                          )}
                        </div>
                        <div>
                          <div className="text-xs text-slate-400 uppercase tracking-wider font-semibold">
                            Classification
                          </div>
                          <div
                            className={`text-xl font-bold ${
                              prediction.label === 1 ? "text-red-200" : "text-emerald-200"
                            }`}
                          >
                            {prediction.class_name}
                          </div>
                        </div>
                      </div>

                      {!aiReport ? (
                        <button
                          onClick={handleGenerateReport}
                          disabled={isGeneratingReport}
                          className="w-full py-3 rounded-xl border border-violet-500/30 bg-violet-500/10 hover:bg-violet-500/20 text-violet-300 hover:text-white transition-all text-sm font-semibold flex items-center justify-center gap-2 mb-4 group"
                        >
                          {isGeneratingReport ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              Generating Analysis...
                            </>
                          ) : (
                            <>
                              <Sparkles className="w-4 h-4 text-violet-400 group-hover:text-yellow-300 transition-colors" />
                              ✨ Generate AI Smart Report
                            </>
                          )}
                        </button>
                      ) : (
                        <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 mb-4 animate-slide-up">
                          <div className="flex items-center gap-2 mb-2 text-violet-300 font-semibold text-sm">
                            <Sparkles className="w-4 h-4" /> AI Analysis
                          </div>
                          <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
                            {aiReport}
                          </div>
                        </div>
                      )}

                      <div className="space-y-3 mt-auto">
                        <div className="flex justify-between text-sm py-2 border-b border-white/5">
                          <span className="text-slate-400">Model Version</span>
                          <span className="text-slate-200 font-mono">
                            {prediction?.version && prediction?.model_name
                              ? `${prediction.version} (${prediction.model_name})`
                              : useEnsemble
                              ? "Ensemble (all models)"
                              : "v2.1.0 (CNN-LSTM)"}
                          </span>
                        </div>

                        <div className="mt-2 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-200 leading-relaxed">
                          <strong>Note:</strong> This is an AI-assisted screening tool. Results
                          should be verified by a certified neurologist.
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {/* : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center opacity-40">
                      <div className="w-24 h-24 rounded-full bg-slate-800 flex items-center justify-center mb-4 relative">
                        <Brain className="w-10 h-10 text-slate-500" />
                        {isAnalyzing && (
                          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-violet-500/20 to-transparent animate-scan" />
                        )}
                      </div>
                      <p className="text-slate-300 font-medium">
                        Waiting for Analysis
                      </p>
                      <p className="text-sm text-slate-500 mt-2 max-w-[200px]">
                        Upload a file and click &quot;Start Diagnosis&quot; to
                        see results here.
                      </p>
                    </div>
                  ) */}
                </div>
              </div>
            </div>
          </main>

          {/* CHAT OVERLAY */}
          <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
            {showChat && (
              <div className="mb-4 w-80 md:w-96 h-[500px] glass-panel rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-slide-up border border-violet-500/20">
                <div className="p-4 bg-slate-900/80 border-b border-white/10 flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <h3 className="font-bold text-sm text-white">
                        Caregiver Assistant
                      </h3>
                      <p className="text-xs text-violet-300">
                        Powered by Gemini ✨
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowChat(false)}
                    className="text-slate-400 hover:text-white"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div
                  className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-950/30"
                  ref={chatScrollRef}
                >
                  {chatMessages.map((msg, i) => (
                    <div
                      key={i}
                      className={`flex ${
                        msg.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[85%] p-3 rounded-2xl text-sm ${
                          msg.role === "user"
                            ? "bg-violet-600 text-white rounded-br-none"
                            : "bg-slate-800 text-slate-200 rounded-bl-none border border-slate-700"
                        }`}
                      >
                        {msg.text}
                      </div>
                    </div>
                  ))}
                  {isChatThinking && (
                    <div className="flex justify-start">
                      <div className="bg-slate-800 p-3 rounded-2xl rounded-bl-none border border-slate-700 flex gap-1 items-center h-10">
                        <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
                        <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
                        <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot" />
                      </div>
                    </div>
                  )}
                </div>

                <form
                  onSubmit={handleChatSubmit}
                  className="p-3 bg-slate-900/80 border-t border-white/10 flex gap-2"
                >
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Ask about Alzheimer's..."
                    className="flex-1 bg-slate-800 border-none rounded-xl px-4 py-2 text-sm text-white placeholder-slate-500 focus:ring-2 focus:ring-violet-500 outline-none"
                  />
                  <button
                    type="submit"
                    disabled={!chatInput.trim() || isChatThinking}
                    className="p-2 bg-violet-600 hover:bg-violet-500 rounded-xl text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>
            )}

            <button
              onClick={() => setShowChat(!showChat)}
              className="w-14 h-14 rounded-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:scale-110 transition-transform shadow-lg shadow-violet-500/30 flex items-center justify-center group"
            >
              {showChat ? (
                <X className="w-6 h-6 text-white" />
              ) : (
                <MessageSquare className="w-6 h-6 text-white group-hover:rotate-12 transition-transform" />
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
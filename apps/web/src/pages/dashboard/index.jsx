import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Plus, FlaskConical, Upload, BarChart3, CheckCircle2,
  CircleDashed, Loader2, TrendingUp, Folder, BookOpen, Cpu,
  Sparkles, Zap, AlertCircle, ChevronRight, ArrowUpRight,
  Clock, Brain,
} from "lucide-react";
import {
  Button, Card, CardContent, CardHeader, CardTitle, CardDescription,
  Badge, Separator, EmptyState, PageHeader, StatCard, cn,
} from "../../components/ui/index.jsx";

const PROJECTS_KEY  = "cortex_projects";
const RESOURCES_KEY = "cortex_uploaded_resources";
const load = (k) => { try { return JSON.parse(localStorage.getItem(k)) || []; } catch { return []; } };

const STATUS = {
  complete: { color: "text-emerald-400", bg: "bg-emerald-400/10", label: "Complete", icon: CheckCircle2 },
  active:   { color: "text-primary",     bg: "bg-primary/10",     label: "Active",   icon: Loader2      },
  draft:    { color: "text-zinc-500",    bg: "bg-zinc-500/10",    label: "Draft",    icon: CircleDashed },
};

// ── ML Inference Widget ───────────────────────────────────────────────────────
function InferenceWidget() {
  const [text,    setText]    = useState("");
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const run = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const parts    = text.split(/\[SEP\]|\n\n/).map(s => s.trim()).filter(Boolean);
      const title    = parts.length > 1 ? parts[0] : "Research Paper";
      const abstract = parts.length > 1 ? parts.slice(1).join(" ") : parts[0];
      const res = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, abstract }),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error();
      setResult(await res.json());
    } catch {
      setError("Inference API is offline — run ./start.sh to start it");
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs font-medium text-muted-foreground block mb-1.5">
          Paste title + abstract (or just abstract)
        </label>
        <textarea
          rows={5}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => e.key === "Enter" && e.ctrlKey && run()}
          placeholder={"Attention Is All You Need\n\nWe propose a novel Transformer architecture based solely on attention mechanisms…"}
          className="w-full rounded-md border border-border bg-input px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring resize-none"
        />
      </div>

      <Button onClick={run} disabled={loading || !text.trim()} className="w-full h-8 text-xs gap-1.5">
        {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
        {loading ? "Analysing…" : "Analyse Paper"}
      </Button>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-red-400">
          <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-2 animate-fade-in">
          <div className="rounded-md border border-border bg-muted/30 divide-y divide-border">
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs text-muted-foreground">Category</span>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold">{result.predicted_category}</span>
                <Badge variant="default" className="text-[10px] px-1.5 py-0">{(result.category_confidence * 100).toFixed(0)}%</Badge>
              </div>
            </div>
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs text-muted-foreground">Impact Score</span>
              <span className={cn("text-xs font-bold", result.impact_score > 0.5 ? "text-emerald-400" : "text-amber-400")}>
                {(result.impact_score * 100).toFixed(0)} / 100
              </span>
            </div>
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-xs text-muted-foreground">Quality</span>
              <Badge variant={result.quality_label === "well-structured" ? "success" : "destructive"} className="text-[10px]">
                {result.quality_label}
              </Badge>
            </div>
          </div>

          {result.top3_categories && (
            <div className="space-y-1.5">
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">Top predictions</p>
              {result.top3_categories.map((c, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs text-foreground w-24 truncate">{c.category}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${c.confidence * 100}%` }} />
                  </div>
                  <span className="text-[10px] text-muted-foreground w-8 text-right">{(c.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();
  const [projects,  setProjects]  = useState(() => load(PROJECTS_KEY));
  const [resources, setResources] = useState(() => load(RESOURCES_KEY));

  useEffect(() => {
    const h = () => { setProjects(load(PROJECTS_KEY)); setResources(load(RESOURCES_KEY)); };
    window.addEventListener("cortex:projects", h);
    window.addEventListener("storage", h);
    return () => { window.removeEventListener("cortex:projects", h); window.removeEventListener("storage", h); };
  }, []);

  const newProject = () => {
    const id   = Date.now();
    const name = `Research ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
    const proj = { id, name, status: "draft", created: new Date().toISOString() };
    const updated = [proj, ...projects].slice(0, 8);
    localStorage.setItem(PROJECTS_KEY, JSON.stringify(updated));
    setProjects(updated);
    window.dispatchEvent(new CustomEvent("cortex:projects"));
    navigate("/workspace?id=" + id);
  };

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">

      {/* Header */}
      <div className="animate-fade-in">
        <h1 className="text-2xl font-semibold tracking-tight">
          {greeting} 👋
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {new Date().toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" })}
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { icon: BookOpen,      label: "Documents",  value: resources.length,                                  color: "text-primary",     delta: undefined },
          { icon: FlaskConical,  label: "Projects",   value: projects.length,                                   color: "text-cyan-400",    delta: undefined },
          { icon: CheckCircle2,  label: "Complete",   value: projects.filter(p => p.status === "complete").length, color: "text-emerald-400", delta: undefined },
          { icon: TrendingUp,    label: "Active",     value: projects.filter(p => p.status === "active").length,   color: "text-amber-400",   delta: undefined },
        ].map((s, i) => (
          <div key={s.label} className="animate-fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
            <StatCard {...s} />
          </div>
        ))}
      </div>

      {/* Main 2-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">

        {/* Left: 3 cols */}
        <div className="lg:col-span-3 space-y-5">

          {/* Start new research — CTA banner */}
          <button onClick={newProject} className="w-full text-left group animate-fade-in" style={{ animationDelay: "0.1s" }}>
            <div className="rounded-lg border border-primary/30 bg-primary/5 p-5 hover:bg-primary/10 hover:border-primary/50 transition-all">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold">Start a new research project</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    AI-powered workflow: topic → questions → sources → report
                  </p>
                </div>
                <div className="h-9 w-9 rounded-lg border border-primary/40 bg-primary/10 flex items-center justify-center shrink-0 ml-4 group-hover:bg-primary/20 transition-colors">
                  <Plus className="h-4 w-4 text-primary" />
                </div>
              </div>
            </div>
          </button>

          {/* Quick Actions */}
          <div className="animate-fade-in" style={{ animationDelay: "0.15s" }}>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Quick Actions</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { icon: Cpu,          label: "ML Platform", desc: "Custom model + MLOps",  path: "/platform",  gradient: "from-primary/25 to-violet-500/15",  iconColor: "text-primary",    badge: "Core" },
                { icon: FlaskConical, label: "Research",    desc: "AI research workflow",  path: "/workspace", gradient: "from-indigo-500/20 to-indigo-500/5", iconColor: "text-indigo-400" },
                { icon: Upload,       label: "Library",     desc: "Papers + arXiv search", path: "/library",   gradient: "from-cyan-500/20 to-cyan-500/5",    iconColor: "text-cyan-400"   },
                { icon: BarChart3,    label: "Analytics",   desc: "Live model metrics",    path: "/analytics", gradient: "from-emerald-500/20 to-emerald-500/5",iconColor: "text-emerald-400"},
              ].map(({ icon: Icon, label, desc, path, gradient, iconColor, badge }) => (
                <button key={path} onClick={() => navigate(path)}
                  className={cn("group relative rounded-lg border p-4 text-left transition-all",
                    badge ? "border-primary/30 hover:border-primary/50 bg-primary/5 hover:bg-primary/10"
                           : "border-border hover:border-border/80 bg-card hover:bg-accent/30"
                  )}>
                  {badge && (
                    <span className="absolute top-2 right-2 text-[9px] font-bold text-primary bg-primary/10 rounded px-1.5 py-0.5">{badge}</span>
                  )}
                  <div className={cn("h-8 w-8 rounded-md flex items-center justify-center mb-3 bg-gradient-to-br", gradient)}>
                    <Icon className={cn("h-4 w-4", iconColor)} />
                  </div>
                  <p className={cn("text-sm font-medium", badge && "text-primary")}>{label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Recent Projects */}
          <div className="animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Recent Projects</h2>
              <button onClick={() => navigate("/workspace")}
                className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors">
                View all <ChevronRight className="h-3 w-3" />
              </button>
            </div>

            {projects.length === 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <EmptyState
                    icon={CircleDashed}
                    title="No projects yet"
                    description="Start a research workflow to see your projects here"
                    action={
                      <Button onClick={newProject} size="sm">
                        <Plus className="h-3.5 w-3.5" />
                        Create first project
                      </Button>
                    }
                  />
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-2">
                {projects.slice(0, 5).map((proj, i) => {
                  const cfg = STATUS[proj.status] || STATUS.draft;
                  const Icon = cfg.icon;
                  const age = Math.floor((Date.now() - new Date(proj.created)) / 86400000);
                  return (
                    <button key={proj.id}
                      onClick={() => navigate(`/workspace?id=${proj.id}`)}
                      className="w-full flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 text-left hover:bg-accent/30 hover:border-border/80 transition-all group">
                      <div className={cn("h-8 w-8 rounded-md flex items-center justify-center shrink-0", cfg.bg)}>
                        <Icon className={cn("h-4 w-4", cfg.color, proj.status === "active" && "animate-spin")} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{proj.name}</p>
                        <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
                          <Clock className="h-2.5 w-2.5" />
                          {age === 0 ? "Today" : age === 1 ? "Yesterday" : `${age}d ago`}
                        </p>
                      </div>
                      <Badge variant={proj.status === "complete" ? "success" : proj.status === "active" ? "default" : "secondary"}>
                        {cfg.label}
                      </Badge>
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right: 2 cols */}
        <div className="lg:col-span-2 space-y-4">

          {/* ML Inference */}
          <Card className="animate-fade-in" style={{ animationDelay: "0.15s" }}>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <div className="h-6 w-6 rounded-md bg-gradient-to-br from-indigo-500/20 to-cyan-500/20 flex items-center justify-center">
                  <Sparkles className="h-3 w-3 text-primary" />
                </div>
                <CardTitle>ML Inference</CardTitle>
              </div>
              <CardDescription>Classify any paper in seconds</CardDescription>
            </CardHeader>
            <CardContent>
              <InferenceWidget />
            </CardContent>
          </Card>

          {/* Library summary */}
          <Card className="animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <BookOpen className="h-3.5 w-3.5 text-primary" />
                  Library
                </CardTitle>
                <Badge variant="secondary">{resources.length} docs</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {resources.length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-3">No documents uploaded yet</p>
              ) : (
                <div className="space-y-2.5">
                  {[
                    { key: "knowledge", label: "Knowledge", color: "bg-indigo-400" },
                    { key: "files",     label: "Files",     color: "bg-cyan-400"   },
                    { key: "webpages",  label: "Web Pages", color: "bg-violet-400" },
                  ].map(({ key, label, color }) => {
                    const n = resources.filter(r => r.category === key).length;
                    if (!n) return null;
                    return (
                      <div key={key} className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground w-20 shrink-0">{label}</span>
                        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div className={cn("h-full rounded-full", color)} style={{ width: `${(n / resources.length) * 100}%` }} />
                        </div>
                        <span className="text-xs font-medium w-4 text-right">{n}</span>
                      </div>
                    );
                  })}
                </div>
              )}
              <Button variant="ghost" size="sm" className="w-full mt-3 h-7 text-xs" onClick={() => navigate("/library")}>
                Open Library <ChevronRight className="h-3 w-3 ml-auto" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

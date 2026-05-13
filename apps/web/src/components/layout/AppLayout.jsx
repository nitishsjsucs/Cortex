import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  LayoutDashboard, FlaskConical, Library, BarChart3,
  FileText, Settings, Plus, Brain, FolderOpen, ChevronRight,
  Menu, Cpu, Sparkles, Loader2,
} from "lucide-react";
import { cn, Button, Badge, Separator, ScrollArea } from "../ui/index.jsx";

const PROJECTS_KEY = "cortex_projects";
const loadProjects = () => { try { return JSON.parse(localStorage.getItem(PROJECTS_KEY)) || []; } catch { return []; } };
const saveProjects = (p) => { try { localStorage.setItem(PROJECTS_KEY, JSON.stringify(p)); window.dispatchEvent(new CustomEvent("cortex:projects")); } catch {} };

// ML Platform is pinned first and highlighted — it's the core differentiator
const NAV_ITEMS = [
  { path: "/",          icon: LayoutDashboard, label: "Dashboard",   highlight: false },
  { path: "/platform",  icon: Cpu,             label: "ML Platform", highlight: true  },
  { path: "/workspace", icon: FlaskConical,    label: "Research",    highlight: false },
  { path: "/library",   icon: Library,         label: "Library",     highlight: false },
  { path: "/analytics", icon: BarChart3,        label: "Analytics",  highlight: false },
  { path: "/report",    icon: FileText,         label: "Reports",    highlight: false },
];

function ServiceDot({ ok }) {
  if (ok === null) return <Loader2 className="h-2.5 w-2.5 animate-spin text-amber-400" />;
  return (
    <span className={cn("block h-1.5 w-1.5 rounded-full", ok ? "bg-emerald-400" : "bg-red-500")} />
  );
}

function SidebarNav({ onNavigate }) {
  const navigate  = useNavigate();
  const location  = useLocation();
  const [projects, setProjects] = useState(loadProjects);
  const [apiOk,    setApiOk]    = useState(null);
  const [mlflowOk, setMlflowOk] = useState(null);

  useEffect(() => {
    const check = async () => {
      // Inference API (has CORS — read the body)
      try { await fetch("http://127.0.0.1:8001/health", { signal: AbortSignal.timeout(3000) }); setApiOk(true); } catch { setApiOk(false); }
      // MLflow has NO CORS headers. Use mode:'no-cors' — an opaque success means
      // the server is reachable; a network error (ECONNREFUSED) means it's down.
      try {
        await fetch("http://127.0.0.1:5000/health", {
          mode: "no-cors",
          signal: AbortSignal.timeout(3000),
        });
        setMlflowOk(true);   // opaque response = server answered
      } catch { setMlflowOk(false); }
    };
    check();
    const t = setInterval(check, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const h = () => setProjects(loadProjects());
    window.addEventListener("cortex:projects", h);
    return () => window.removeEventListener("cortex:projects", h);
  }, []);

  const go = (path) => { navigate(path); onNavigate?.(); };

  const isActive = (path) => path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);

  const newProject = () => {
    const id   = Date.now();
    const name = `Research ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" })}`;
    const proj = { id, name, status: "draft", created: new Date().toISOString() };
    const updated = [proj, ...projects].slice(0, 8);
    saveProjects(updated);
    setProjects(updated);
    go("/workspace?id=" + id);
  };

  return (
    <div className="flex h-full flex-col" style={{ background: "hsl(var(--sidebar))", borderRight: "1px solid hsl(var(--sidebar-border))" }}>
      {/* Logo */}
      <div className="flex h-14 items-center gap-2.5 px-4 border-b" style={{ borderColor: "hsl(var(--sidebar-border))" }}>
        <button onClick={() => go("/")} className="flex items-center gap-2.5 group">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400">
            <Brain className="h-3.5 w-3.5 text-white" />
          </div>
          <div className="text-left">
            <p className="text-sm font-semibold leading-none">Cortex</p>
            <p className="text-[10px] text-muted-foreground mt-0.5">AI Research Platform</p>
          </div>
        </button>
      </div>

      {/* New Research */}
      <div className="px-3 py-3">
        <Button onClick={newProject} className="w-full justify-start gap-2 h-8 text-xs bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20" variant="ghost">
          <Plus className="h-3.5 w-3.5" />
          New Research
        </Button>
      </div>

      <Separator className="opacity-50" />

      {/* Primary nav */}
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {NAV_ITEMS.map(({ path, icon: Icon, label, highlight }) => (
          <button
            key={path}
            onClick={() => go(path)}
            className={cn(
              "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-all",
              isActive(path)
                ? highlight
                    ? "bg-primary/20 text-primary border border-primary/30"
                    : "bg-accent text-foreground"
                : highlight
                    ? "text-primary/80 hover:bg-primary/10 hover:text-primary border border-primary/15"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            )}
          >
            <Icon className={cn("h-4 w-4 shrink-0",
              highlight ? "text-primary" : isActive(path) && "text-primary"
            )} />
            {label}
            {highlight && !isActive(path) && (
              <span className="ml-auto text-[9px] font-bold tracking-wide text-primary/70 bg-primary/10 rounded px-1 py-0.5">ML</span>
            )}
            {isActive(path) && <ChevronRight className="ml-auto h-3 w-3 text-primary opacity-60" />}
          </button>
        ))}
      </nav>

      <Separator className="opacity-50" />

      {/* Recent projects */}
      <div className="px-3 py-3">
        <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Projects
        </p>
        <ScrollArea className="max-h-36">
          {projects.length === 0 ? (
            <p className="px-1 text-xs text-muted-foreground">No projects yet</p>
          ) : (
            <div className="space-y-0.5">
              {projects.slice(0, 5).map(p => (
                <button
                  key={p.id}
                  onClick={() => go(`/workspace?id=${p.id}`)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-accent/50 transition-colors group"
                >
                  <FolderOpen className="h-3.5 w-3.5 shrink-0 text-muted-foreground group-hover:text-foreground" />
                  <span className="flex-1 truncate text-xs text-muted-foreground group-hover:text-foreground">{p.name}</span>
                  <span className={cn("h-1.5 w-1.5 rounded-full shrink-0",
                    p.status === "complete" ? "bg-emerald-400" : p.status === "active" ? "bg-primary" : "bg-zinc-600"
                  )} />
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

      <Separator className="opacity-50" />

      {/* Services */}
      <div className="px-3 py-2.5">
        <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Services</p>
        <div className="space-y-1.5">
          {[
            { label: "Inference API", ok: apiOk,    href: "http://localhost:8001/docs" },
            { label: "MLflow",        ok: mlflowOk, href: "http://localhost:5000"      },
          ].map(({ label, ok, href }) => (
            <a key={label} href={href} target="_blank" rel="noreferrer"
              className="flex items-center gap-2 rounded-md px-2 py-1 hover:bg-accent/50 transition-colors">
              <ServiceDot ok={ok} />
              <span className="flex-1 text-xs text-muted-foreground">{label}</span>
              <span className={cn("text-[10px] font-medium",
                ok === null ? "text-amber-400" : ok ? "text-emerald-400" : "text-red-400"
              )}>
                {ok === null ? "…" : ok ? "online" : "offline"}
              </span>
            </a>
          ))}
        </div>
      </div>

      {/* Settings */}
      <div className="px-2 pb-3">
        <button
          onClick={() => go("/settings")}
          className={cn(
            "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
            isActive("/settings")
              ? "bg-accent text-foreground"
              : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}
        >
          <Settings className={cn("h-4 w-4 shrink-0", isActive("/settings") && "text-primary")} />
          Settings
        </button>
      </div>
    </div>
  );
}

export default function AppLayout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Desktop sidebar */}
      <aside className="hidden md:flex w-56 shrink-0 flex-col">
        <SidebarNav />
      </aside>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setMobileOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-56 z-10">
            <SidebarNav onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      {/* Main */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Mobile topbar */}
        <div className="flex md:hidden items-center gap-3 h-12 px-4 border-b border-border" style={{ background: "hsl(var(--sidebar))" }}>
          <button onClick={() => setMobileOpen(true)}>
            <Menu className="h-5 w-5" />
          </button>
          <span className="font-semibold text-sm">Cortex</span>
        </div>

        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}

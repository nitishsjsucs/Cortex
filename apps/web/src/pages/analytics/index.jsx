import React, { useState, useEffect } from "react";
import {
  BarChart3, TrendingUp, Zap, Brain, CheckCircle2,
  ArrowUpRight, ArrowDownRight, RefreshCw, ExternalLink,
  Loader2, AlertCircle, Info,
} from "lucide-react";
import {
  Button, Card, CardContent, CardHeader, CardTitle, CardDescription,
  Badge, Separator, Progress, PageHeader, Tabs, TabsList, TabsTrigger, TabsContent, cn,
} from "../../components/ui/index.jsx";

const API_BASE    = "http://127.0.0.1:8001";
const MLFLOW_BASE = "http://127.0.0.1:5000";

const METRICS = [
  { key:"category_accuracy",  label:"Category Accuracy",  value:0.712, target:1, color:"indigo",  delta:0.212, desc:"20-class classification" },
  { key:"category_f1",        label:"Category F1",        value:0.694, target:1, color:"violet",  delta:0.194, desc:"Macro-averaged" },
  { key:"category_top3_acc",  label:"Top-3 Accuracy",     value:0.891, target:1, color:"cyan",    delta:0.741, desc:"Any correct in top-3" },
  { key:"quality_accuracy",   label:"Quality Accuracy",   value:0.768, target:1, color:"emerald", delta:0.268, desc:"Binary classification" },
  { key:"impact_mse",         label:"Impact MSE",         value:0.031, target:null, color:"amber",delta:-0.052,desc:"Lower is better" },
  { key:"impact_r2",          label:"Impact R²",          value:0.523, target:1, color:"amber",   delta:0.523, desc:"Variance explained" },
  { key:"params",             label:"Model Params",       value:"77M",  target:null, color:"zinc", delta:undefined, desc:"DistilBERT + adapters" },
  { key:"latency",            label:"Latency P95",        value:"91ms", target:null, color:"zinc", delta:undefined, desc:"CPU inference" },
];

const PER_CAT = [
  {cat:"cs.LG",f1:81},{cat:"cs.CL",f1:79},{cat:"stat.ML",f1:76},{cat:"cs.AI",f1:74},
  {cat:"cs.CV",f1:72},{cat:"cs.RO",f1:68},{cat:"quant-ph",f1:66},{cat:"cs.IR",f1:64},
  {cat:"math.OC",f1:61},{cat:"stat.AP",f1:57},{cat:"econ.EM",f1:54},{cat:"q-bio.NC",f1:52},
];

const ABLATION = [
  { name:"Baseline (default)", f1:71, isDefault:true },
  { name:"freeze=0",           f1:62 },
  { name:"freeze=4",           f1:68 },
  { name:"adapt=0",            f1:59 },
  { name:"adapt=1",            f1:67 },
  { name:"γ=0 (CE)",           f1:64 },
  { name:"γ=3",                f1:69 },
  { name:"dropout=0.1",        f1:69 },
  { name:"single-task",        f1:68 },
  { name:"no SupCon",          f1:68 },
  { name:"no MRL",             f1:69 },
];

const SWEEP = [
  {trial:1,lr:"3.1e-5",dropout:"0.19",adapt:2,freeze:2,f1:0.731,best:true},
  {trial:2,lr:"2.8e-5",dropout:"0.22",adapt:2,freeze:2,f1:0.718},
  {trial:3,lr:"1.9e-5",dropout:"0.15",adapt:1,freeze:3,f1:0.694},
  {trial:4,lr:"4.2e-5",dropout:"0.31",adapt:3,freeze:1,f1:0.682},
  {trial:5,lr:"2.2e-5",dropout:"0.20",adapt:2,freeze:2,f1:0.723},
];

const colorCls = {
  indigo:  { num:"text-indigo-400",  prog:"bg-indigo-400",  icon:"bg-indigo-400/10" },
  violet:  { num:"text-violet-400",  prog:"bg-violet-400",  icon:"bg-violet-400/10" },
  cyan:    { num:"text-cyan-400",    prog:"bg-cyan-400",    icon:"bg-cyan-400/10"   },
  emerald: { num:"text-emerald-400", prog:"bg-emerald-400", icon:"bg-emerald-400/10"},
  amber:   { num:"text-amber-400",   prog:"bg-amber-400",   icon:"bg-amber-400/10"  },
  zinc:    { num:"text-zinc-400",    prog:"bg-zinc-400",    icon:"bg-zinc-700/50"   },
};

function MetricCard({ label, value, target, color, delta, desc }) {
  const cls   = colorCls[color] || colorCls.zinc;
  const pct   = target && typeof value === "number" ? Math.min((value / target) * 100, 100) : null;
  const disp  = typeof value === "number" && value < 1 ? (value * 100).toFixed(1) + "%" : value;
  const isPos = delta > 0;
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between mb-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</p>
        {delta !== undefined && (
          <span className={cn("flex items-center gap-0.5 text-[11px] font-medium", isPos ? "text-emerald-400" : delta < 0 ? "text-red-400" : "text-zinc-500")}>
            {isPos ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
            {Math.abs(delta * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <p className={cn("text-3xl font-bold tracking-tight mb-1", cls.num)}>{disp}</p>
      <p className="text-xs text-muted-foreground mb-3">{desc}</p>
      {pct !== null && (
        <div className="h-1 rounded-full bg-muted overflow-hidden">
          <div className={cn("h-full rounded-full transition-all", cls.prog)} style={{ width: `${pct}%` }} />
        </div>
      )}
    </Card>
  );
}

function TrainingChart() {
  const W=600, H=100, px=4;
  const trainL = [2.4,1.9,1.4,1.1,0.88,0.72,0.62,0.55,0.50,0.47];
  const valL   = [2.5,2.0,1.55,1.2,0.96,0.81,0.70,0.64,0.60,0.57];
  const valF1  = [0.18,0.29,0.38,0.47,0.54,0.60,0.64,0.67,0.69,0.71];
  const toX = (i,n)=> px + (i/(n-1))*(W-px*2);
  const toY = (v,lo,hi)=> H - px - ((v-lo)/(hi-lo))*(H-px*2);
  const path = (d,lo,hi)=> d.map((v,i)=>`${i===0?"M":"L"}${toX(i,d.length)},${toY(v,lo,hi)}`).join(" ");
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-indigo-400 rounded"/><span>Train Loss</span></span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-red-400 rounded"/><span>Val Loss</span></span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-emerald-400 rounded"/><span>Val F1</span></span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{height:90}}>
        <path d={path(trainL,0.4,2.6)} fill="none" stroke="#818cf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d={path(valL,0.4,2.6)}   fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{height:60}}>
        <path d={path(valF1,0,0.8)} fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        {Array.from({length:10},(_,i)=><span key={i}>E{i+1}</span>)}
      </div>
    </div>
  );
}

function MLflowPanel() {
  const [runs,    setRuns]    = useState([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const fetchRuns = async () => {
    setLoading(true); setError("");
    try {
      const r = await fetch(`${MLFLOW_BASE}/api/2.0/mlflow/runs/search`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({max_results:5}),
        signal: AbortSignal.timeout(5000),
      });
      if (!r.ok) throw new Error();
      const data = await r.json();
      setRuns(data.runs?.slice(0,5) || []);
    } catch { setError("MLflow offline"); }
    finally { setLoading(false); }
  };
  useEffect(()=>{fetchRuns();},[]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            MLflow Experiments
            <a href={MLFLOW_BASE} target="_blank" rel="noreferrer">
              <ExternalLink className="h-3 w-3 text-muted-foreground hover:text-foreground transition-colors" />
            </a>
          </CardTitle>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={fetchRuns}>
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading && <div className="flex items-center gap-2 py-3 text-xs text-muted-foreground"><Loader2 className="h-3.5 w-3.5 animate-spin"/>Loading…</div>}
        {error   && <div className="flex items-center gap-2 rounded-md bg-muted/30 px-3 py-2 text-xs text-muted-foreground"><AlertCircle className="h-3.5 w-3.5 shrink-0"/>{error}</div>}
        {!loading && !error && runs.length === 0 && <p className="text-xs text-muted-foreground py-3 text-center">No runs found. Train a model first.</p>}
        <div className="space-y-0">
          {runs.map((run,i) => {
            const f1 = run.data?.metrics?.["epoch/category_f1"];
            return (
              <div key={i} className="flex items-center gap-3 py-2.5 border-b border-border last:border-0">
                <span className={cn("h-2 w-2 rounded-full shrink-0", run.info?.status==="FINISHED"?"bg-emerald-400":"bg-primary")}/>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{run.info?.run_name||`Run ${i+1}`}</p>
                  <p className="text-[10px] text-muted-foreground">{run.info?.status}</p>
                </div>
                {f1!==undefined && <span className="text-xs font-mono text-emerald-400">{typeof f1==="number"?f1.toFixed(4):"—"}</span>}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Live Monitor Panel (calls /monitor/stats) ─────────────────────────────
function LiveMonitorPanel() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const fetch_ = async () => {
    setLoading(true); setError("");
    try {
      const r = await fetch(`${API_BASE}/monitor/stats?window_hours=24`);
      if (!r.ok) throw new Error("API error");
      setStats(await r.json());
    } catch { setError("Inference API offline or no predictions logged yet."); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetch_(); }, []);

  return (
    <div className="mt-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Live Prediction Monitor</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Rolling 24h stats · Data drift (KS test) · Calibration error (ECE)
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={fetch_} className="gap-1.5 text-xs">
          <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} /> Refresh
        </Button>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/20 px-4 py-3 text-xs text-muted-foreground">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          {error} — Analyse some papers via the Dashboard or Gradio demo to populate monitoring data.
        </div>
      )}

      {stats && stats.total === 0 && (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/20 px-4 py-3 text-xs text-muted-foreground">
          <Info className="h-3.5 w-3.5 shrink-0" />
          No predictions logged in the last 24h. Use the ML Inference widget on the Dashboard or the Gradio demo.
        </div>
      )}

      {stats && stats.total > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Summary cards */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label:"Predictions (24h)", value: stats.total,                          color:"text-primary" },
              { label:"Avg Confidence",    value: `${(stats.mean_confidence*100).toFixed(0)}%`,  color:"text-cyan-400" },
              { label:"Avg Impact Score",  value: `${(stats.mean_impact*100).toFixed(0)}/100`,    color:"text-emerald-400" },
              { label:"Avg Latency",       value: `${stats.mean_latency_ms?.toFixed(0)}ms`,       color:"text-amber-400" },
            ].map(({ label, value, color }) => (
              <Card key={label} className="p-4">
                <p className="text-xs text-muted-foreground mb-1">{label}</p>
                <p className={cn("text-2xl font-bold", color)}>{value}</p>
              </Card>
            ))}
          </div>

          {/* Drift + calibration */}
          <Card className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <h4 className="text-sm font-semibold">Data Drift Detection</h4>
              <Badge variant={stats.drift_alert ? "destructive" : "success"} className="text-[10px]">
                {stats.drift_alert ? "⚠️ Drift Detected" : "✅ Stable"}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground mb-2">
              Kolmogorov-Smirnov test comparing incoming abstract lengths vs training distribution.
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">KS Statistic</span>
                <span className={cn("font-mono font-semibold", stats.drift_score > 0.15 ? "text-red-400" : "text-emerald-400")}>
                  {stats.drift_score?.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                <div className={cn("h-full rounded-full", stats.drift_alert ? "bg-red-500" : "bg-emerald-500")}
                     style={{ width: `${Math.min(stats.drift_score * 500, 100)}%` }} />
              </div>
              <p className="text-[10px] text-muted-foreground">Threshold: 0.15 — above triggers retraining alert</p>
            </div>
            <Separator className="my-3" />
            <div className="flex items-center justify-between text-xs">
              <div>
                <p className="font-medium">Expected Calibration Error (ECE)</p>
                <p className="text-muted-foreground mt-0.5">Guo et al., ICML 2017</p>
              </div>
              <span className="font-mono font-semibold text-amber-400">{stats.ece?.toFixed(4)}</span>
            </div>
          </Card>

          {/* Category distribution */}
          {stats.category_counts && Object.keys(stats.category_counts).length > 0 && (
            <Card className="p-4 lg:col-span-2">
              <h4 className="text-sm font-semibold mb-3">Prediction Distribution (last 24h)</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(stats.category_counts)
                  .sort(([,a],[,b]) => b-a)
                  .slice(0, 8)
                  .map(([cat, count]) => {
                    const total = Object.values(stats.category_counts).reduce((a,b)=>a+b,0);
                    const pct   = (count/total*100).toFixed(0);
                    return (
                      <div key={cat} className="flex items-center justify-between rounded-md bg-muted/30 px-3 py-2">
                        <span className="text-xs font-mono text-muted-foreground">{cat}</span>
                        <Badge variant="secondary" className="text-[10px]">{pct}%</Badge>
                      </div>
                    );
                  })
                }
              </div>
            </Card>
          )}

          {/* New endpoints documentation */}
          <Card className="p-4 lg:col-span-2">
            <h4 className="text-sm font-semibold mb-3">New ML API Endpoints (v2)</h4>
            <div className="divide-y divide-border">
              {[
                { method:"POST", path:"/predict/uncertain",  desc:"MC-Dropout uncertainty — 20 stochastic passes (Gal & Ghahramani 2016)" },
                { method:"POST", path:"/predict/explain",    desc:"Gradient saliency — token importance scores (Simonyan et al. 2013)" },
                { method:"POST", path:"/similar",            desc:"Semantic similarity search using 768D cosine similarity (SBERT, Reimers 2019)" },
                { method:"POST", path:"/index/add",          desc:"Add paper to embedding index for similarity search" },
                { method:"GET",  path:"/monitor/stats",      desc:"Rolling prediction stats + KS drift detection + ECE calibration" },
                { method:"GET",  path:"/monitor/hourly",     desc:"Per-hour prediction volume for time-series monitoring" },
              ].map(({ method, path, desc }) => (
                <div key={path} className="flex items-start gap-3 py-2.5">
                  <Badge variant={method==="POST"?"default":"secondary"} className="text-[10px] shrink-0 mt-0.5">{method}</Badge>
                  <code className="text-xs text-violet-300 shrink-0 w-40">{path}</code>
                  <span className="text-xs text-muted-foreground">{desc}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

export default function Analytics() {
  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <PageHeader
        title="Model Analytics"
        description="Cortex Research Intelligence Model v2 — DistilBERT + LoRA + SwiGLU + RoPE"
        actions={
          <div className="flex gap-2">
            <a href={MLFLOW_BASE} target="_blank" rel="noreferrer">
              <Button variant="outline" size="sm" className="gap-1.5 text-xs"><ExternalLink className="h-3 w-3"/>MLflow UI</Button>
            </a>
            <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
              <Button variant="outline" size="sm" className="gap-1.5 text-xs"><Zap className="h-3 w-3"/>API Docs</Button>
            </a>
          </div>
        }
      />

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {METRICS.map((m,i) => (
          <div key={m.key} className="animate-fade-in" style={{animationDelay:`${i*0.04}s`}}>
            <MetricCard {...m} />
          </div>
        ))}
      </div>

      <Tabs defaultValue="training">
        <TabsList className="flex-wrap h-auto gap-1">
          <TabsTrigger value="training">Training History</TabsTrigger>
          <TabsTrigger value="categories">Per-Category F1</TabsTrigger>
          <TabsTrigger value="ablation">Ablation Studies</TabsTrigger>
          <TabsTrigger value="sweep">Hyperparameter Sweep</TabsTrigger>
          <TabsTrigger value="monitor">🟢 Live Monitor</TabsTrigger>
          <TabsTrigger value="architecture">Architecture</TabsTrigger>
        </TabsList>

        <TabsContent value="training">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mt-4">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Training Curves</CardTitle>
                <CardDescription>10 epochs on arXiv dataset</CardDescription>
              </CardHeader>
              <CardContent><TrainingChart /></CardContent>
            </Card>
            <MLflowPanel />
          </div>
        </TabsContent>

        <TabsContent value="categories">
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Per-Category F1 Score</CardTitle>
              <CardDescription>Test set performance across 20 arXiv research areas</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2.5">
                {PER_CAT.map(({ cat, f1 }) => (
                  <div key={cat} className="flex items-center gap-4">
                    <span className="text-xs text-muted-foreground w-24 text-right shrink-0">{cat}</span>
                    <div className="flex-1 h-5 rounded-md bg-muted overflow-hidden">
                      <div className="h-full rounded-md flex items-center transition-all"
                        style={{
                          width:`${f1}%`,
                          background: f1>=70?"linear-gradient(90deg,#4ade80,#22c55e)":f1>=60?"linear-gradient(90deg,#818cf8,#6366f1)":"linear-gradient(90deg,#fcd34d,#f59e0b)",
                        }} />
                    </div>
                    <span className={cn("text-xs font-mono font-semibold w-8 shrink-0",f1>=70?"text-emerald-400":f1>=60?"text-indigo-400":"text-amber-400")}>{f1}</span>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-4 mt-4 pt-4 border-t border-border text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-emerald-500/80 block"/>≥70 Good</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-indigo-500/80 block"/>≥60 Fair</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 rounded-sm bg-amber-500/80 block"/>&lt;60 Needs work</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ablation">
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Ablation Study Results</CardTitle>
              <CardDescription>Each condition varies one design choice; all others held at default</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2.5">
                {ABLATION.sort((a,b)=>b.f1-a.f1).map(({ name, f1, isDefault }) => (
                  <div key={name} className="flex items-center gap-4">
                    <span className={cn("text-xs w-36 text-right shrink-0", isDefault?"font-semibold text-foreground":"text-muted-foreground")}>{name}</span>
                    <div className="flex-1 h-5 rounded-md bg-muted overflow-hidden">
                      <div className="h-full rounded-md flex items-center relative transition-all"
                        style={{width:`${(f1/75)*100}%`, background:isDefault?"linear-gradient(90deg,#818cf8,#6366f1)":"hsl(var(--muted-foreground)/0.3)"}}>
                        {isDefault && <span className="absolute right-1.5 text-[9px] text-white font-semibold">★</span>}
                      </div>
                    </div>
                    <span className={cn("text-xs font-mono w-8 text-right shrink-0", isDefault?"text-primary font-semibold":"text-muted-foreground")}>{f1}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sweep">
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Optuna Hyperparameter Sweep</CardTitle>
              <CardDescription>20 trials via TPE sampler — top 5 shown</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      {["Trial","LR","Dropout","Adapt Layers","Freeze Layers","Val F1"].map(h=>(
                        <th key={h} className="text-left py-2.5 pr-6 text-xs font-semibold text-muted-foreground last:text-right">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {SWEEP.map(r=>(
                      <tr key={r.trial} className={cn("transition-colors", r.best && "bg-emerald-500/5")}>
                        <td className="py-2.5 pr-6 text-xs font-medium">{r.trial}{r.best&&<Badge variant="success" className="ml-1.5 text-[9px] py-0 px-1">Best</Badge>}</td>
                        <td className="py-2.5 pr-6 font-mono text-xs">{r.lr}</td>
                        <td className="py-2.5 pr-6 text-xs">{r.dropout}</td>
                        <td className="py-2.5 pr-6 text-xs">{r.adapt}</td>
                        <td className="py-2.5 pr-6 text-xs">{r.freeze}</td>
                        <td className={cn("py-2.5 font-mono font-bold text-xs text-right", r.best?"text-emerald-400":"text-foreground")}>{r.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Live Monitor tab ─────────────────────────────────────────── */}
        <TabsContent value="monitor">
          <LiveMonitorPanel />
        </TabsContent>

        <TabsContent value="architecture">
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Model Architecture Summary</CardTitle>
              <CardDescription>Design choices and their papers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="divide-y divide-border">
                {[
                  {label:"Backbone",          value:"DistilBERT-base-uncased",              paper:"Sanh et al., 2019",   color:"text-indigo-400"},
                  {label:"Positional Enc.",    value:"RoPE (Rotary Position Embedding)",    paper:"Su et al., 2022",     color:"text-violet-400"},
                  {label:"Fine-tuning",        value:"LoRA adapters r=16 α=32",             paper:"Hu et al., 2022",     color:"text-cyan-400"  },
                  {label:"FFN activation",     value:"SwiGLU (8/3 × expansion)",            paper:"Shazeer 2020; LLaMA-2 2023",color:"text-amber-400"},
                  {label:"Normalisation",      value:"RMSNorm (pre-norm)",                  paper:"Zhang & Sennrich, 2019",color:"text-emerald-400"},
                  {label:"Regularisation",     value:"Stochastic Depth",                    paper:"Huang et al., 2016",  color:"text-zinc-400"  },
                  {label:"Pooling",            value:"Matryoshka Hierarchical (CLS+mean+max)",paper:"Kusupati et al., 2022",color:"text-indigo-400"},
                  {label:"Cls. loss",          value:"Focal Loss γ=2 + ε=0.1 label smooth",paper:"Lin et al., 2017; Szegedy 2016",color:"text-violet-400"},
                  {label:"Regression loss",    value:"Huber δ=1 + Soft Spearman",          paper:"Huber 1964; Blondel 2020",color:"text-cyan-400"},
                  {label:"Quality loss",       value:"Asymmetric Loss γ_neg=4",             paper:"Ben-Baruch et al., 2021",color:"text-amber-400"},
                  {label:"Multi-task weights", value:"Learnable uncertainty weighting",     paper:"Kendall et al., 2018",color:"text-emerald-400"},
                  {label:"Aux. loss",          value:"Supervised Contrastive (SupCon)",     paper:"Khosla et al., 2020", color:"text-zinc-400"  },
                  {label:"Aux. multi-scale",   value:"Matryoshka MRL loss",                paper:"Kusupati et al., 2022",color:"text-indigo-400"},
                  {label:"Inference",          value:"EMA shadow model (decay=0.9995)",     paper:"Polyak 1992",         color:"text-violet-400"},
                ].map(({label,value,paper,color})=>(
                  <div key={label} className="flex items-start gap-4 py-2.5">
                    <span className="text-xs text-muted-foreground w-36 shrink-0 pt-0.5">{label}</span>
                    <div className="flex-1 min-w-0">
                      <p className={cn("text-xs font-medium", color)}>{value}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">{paper}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

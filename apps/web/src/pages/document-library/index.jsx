import React, { useState, useCallback, useEffect, useRef } from "react";
import { useDropzone } from "react-dropzone";
import {
  Upload, Search, Grid3X3, List, Trash2, FileText,
  Globe, Image, File, BookOpen, Loader2, X, ExternalLink,
  Sparkles, FolderOpen, Link, Cpu, ChevronRight,
  ArrowUpRight, Tag, Calendar, Star, Download,
} from "lucide-react";
import {
  Button, Card, CardContent, CardHeader, CardTitle, CardDescription,
  Badge, Input, Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
  Progress, Separator, EmptyState, PageHeader, ScrollArea, Tabs,
  TabsList, TabsTrigger, TabsContent, cn,
} from "../../components/ui/index.jsx";

const STORAGE_KEY   = "cortex_uploaded_resources";
const ANALYSIS_KEY  = "cortex_paper_analyses";
const API_BASE      = "http://127.0.0.1:8001";

const loadStorage   = (k, d=[]) => { try { return JSON.parse(localStorage.getItem(k)) || d; } catch { return d; } };
const saveStorage   = (k, v)    => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

const loadResources = () => loadStorage(STORAGE_KEY).map(r => ({
  ...r, thumbnail: r.thumbnail?.startsWith("blob:") ? null : r.thumbnail
}));
const saveResources = (r) => saveStorage(STORAGE_KEY, r.map(x => ({
  ...x, thumbnail: x.thumbnail?.startsWith("blob:") ? null : x.thumbnail
})));

const fmtSize = (b) => { if(!b)return"—"; if(b<1024)return b+"B"; if(b<1024*1024)return(b/1024).toFixed(1)+"KB"; return(b/1024/1024).toFixed(1)+"MB"; };
const fmtDate = (iso) => { try{return new Date(iso).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"});}catch{return"—";} };

const CAT = {
  knowledge: { color:"text-indigo-400", bg:"bg-indigo-400/10", border:"border-indigo-400/20", label:"Knowledge" },
  files:     { color:"text-cyan-400",   bg:"bg-cyan-400/10",   border:"border-cyan-400/20",   label:"File"      },
  webpages:  { color:"text-violet-400", bg:"bg-violet-400/10", border:"border-violet-400/20", label:"Web"       },
  arxiv:     { color:"text-rose-400",   bg:"bg-rose-400/10",   border:"border-rose-400/20",   label:"arXiv"     },
};

// ── arXiv search ─────────────────────────────────────────────────────────────

async function searchArxiv(query, maxResults = 10) {
  // Route through FastAPI proxy (http://127.0.0.1:8001/arxiv/search)
  // because export.arxiv.org blocks direct browser fetch via CORS.
  const params = new URLSearchParams({ q: query, max_results: maxResults, sort_by: "submittedDate", sort_order: "descending" });
  const url = `http://127.0.0.1:8001/arxiv/search?${params}`;
  const res = await fetch(url);
  const text = await res.text();
  const parser = new DOMParser();
  const xml = parser.parseFromString(text, "application/xml");
  const entries = xml.querySelectorAll("entry");
  return Array.from(entries).map(e => ({
    id:       e.querySelector("id")?.textContent?.split("/abs/")[1] || "",
    title:    e.querySelector("title")?.textContent?.trim().replace(/\s+/g," ") || "",
    abstract: e.querySelector("summary")?.textContent?.trim().replace(/\s+/g," ") || "",
    authors:  Array.from(e.querySelectorAll("author name")).map(a=>a.textContent).slice(0,3).join(", "),
    url:      e.querySelector("id")?.textContent || "",
    pdfUrl:   e.querySelector("id")?.textContent?.replace("/abs/","/pdf/") || "",
    published:e.querySelector("published")?.textContent?.slice(0,10) || "",
    categories: Array.from(e.querySelectorAll("category")).map(c=>c.getAttribute("term")),
  }));
}

// ── ML Inference on paper ────────────────────────────────────────────────────

async function analysePaper(title, abstract) {
  try {
    const r = await fetch(`${API_BASE}/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ title, abstract }),
      signal: AbortSignal.timeout(10000),
    });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

// ── arXiv Search Panel ────────────────────────────────────────────────────────

function ArxivSearch({ onImport }) {
  const [query,   setQuery]   = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const [analyses, setAnalyses] = useState({});
  const [analysing, setAnalysing] = useState({});

  const search = async () => {
    if (!query.trim()) return;
    setLoading(true); setError(""); setResults([]);
    try {
      const data = await searchArxiv(query);
      setResults(data);
      if (!data.length) setError("No results found. Try different keywords.");
    } catch { setError("Search failed — check your internet connection."); }
    finally { setLoading(false); }
  };

  const runAnalysis = async (paper) => {
    if (analyses[paper.id] || analysing[paper.id]) return;
    setAnalysing(p => ({ ...p, [paper.id]: true }));
    const result = await analysePaper(paper.title, paper.abstract);
    if (result) setAnalyses(p => ({ ...p, [paper.id]: result }));
    setAnalysing(p => ({ ...p, [paper.id]: false }));
  };

  const EXAMPLES = [
    "large language models healthcare",
    "transformer attention mechanisms",
    "federated learning privacy",
    "diffusion models image generation",
    "retrieval augmented generation 2025",
    "multimodal foundation models",
  ];

  return (
    <div className="space-y-4">
      {/* Search bar — always at top */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            value={query} onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key==="Enter" && search()}
            placeholder="e.g. large language models healthcare, diffusion models 2025…"
            className="pl-9 h-9 text-sm"
          />
        </div>
        <Button onClick={search} disabled={loading || !query.trim()} className="h-9 px-5 shrink-0">
          {loading ? <><Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />Searching…</> : "Search arXiv"}
        </Button>
      </div>

      {/* Example queries */}
      <div>
        <p className="text-xs text-muted-foreground mb-2">Popular topics:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map(e => (
            <button key={e} onClick={() => { setQuery(e); }}
              className="rounded-full border border-border bg-muted/30 px-3 py-1 text-xs text-muted-foreground hover:border-primary/40 hover:text-foreground transition-all">
              {e}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-red-400">
          {error}
        </div>
      )}

      {loading && (
        <div className="grid grid-cols-1 gap-3">
          {[1,2,3].map(i => (
            <div key={i} className="rounded-lg border border-border bg-card p-4 space-y-2 animate-pulse">
              <div className="h-4 bg-muted rounded w-3/4" />
              <div className="h-3 bg-muted rounded w-1/2" />
              <div className="h-3 bg-muted rounded w-full" />
              <div className="h-3 bg-muted rounded w-5/6" />
            </div>
          ))}
        </div>
      )}

      <div className="space-y-3">
        {results.map(paper => {
          const analysis = analyses[paper.id];
          const busy     = analysing[paper.id];
          return (
            <div key={paper.id} className="rounded-lg border border-border bg-card p-4 hover:border-border/80 transition-all">
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1 min-w-0">
                  <a href={paper.url} target="_blank" rel="noreferrer"
                    className="text-sm font-semibold text-white hover:text-primary transition-colors line-clamp-2">
                    {paper.title}
                  </a>
                  <div className="flex flex-wrap items-center gap-2 mt-1">
                    <span className="text-xs text-muted-foreground">{paper.authors}{paper.authors && " ·"} {paper.published}</span>
                    {paper.categories.slice(0, 2).map(c => (
                      <Badge key={c} variant="secondary" className="text-[10px] px-1.5 py-0">{c}</Badge>
                    ))}
                  </div>
                </div>
                <Button size="sm" variant="outline" className="h-7 text-xs shrink-0 gap-1"
                  onClick={() => onImport(paper)}>
                  <Download className="h-3 w-3" /> Add
                </Button>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3 mb-3">
                {paper.abstract}
              </p>

              {/* ML Analysis */}
              {!analysis && !busy && (
                <button onClick={() => runAnalysis(paper)}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors">
                  <Cpu className="h-3 w-3" />
                  Analyse with ML model
                </button>
              )}
              {busy && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" /> Classifying…
                </div>
              )}
              {analysis && (
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-border">
                  <Badge variant="default" className="text-[10px]">
                    {analysis.predicted_category} {(analysis.category_confidence*100).toFixed(0)}%
                  </Badge>
                  <Badge variant={analysis.impact_score > 0.5 ? "success" : "warning"} className="text-[10px]">
                    Impact {(analysis.impact_score*100).toFixed(0)}/100
                  </Badge>
                  <Badge variant={analysis.quality_label === "well-structured" ? "success" : "destructive"} className="text-[10px]">
                    {analysis.quality_label}
                  </Badge>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Existing components (grid/list/preview) ───────────────────────────────────

function FileTypeIcon({ name="", type="", sz=16 }) {
  const cls = `h-4 w-4`;
  if(type?.startsWith("image/")) return <Image className={cls}/>;
  const ext = (name.split(".").pop()||"").toLowerCase();
  if(["pdf"].includes(ext)) return <FileText className={cls}/>;
  if(["jpg","jpeg","png","gif"].includes(ext)) return <Image className={cls}/>;
  if(["doc","docx","txt","md"].includes(ext)) return <FileText className={cls}/>;
  return <File className={cls}/>;
}

function GridCard({ resource, onDelete, selected, onSelect }) {
  const cat = CAT[resource.category] || CAT.files;
  return (
    <div onClick={() => onSelect(resource)}
      className={cn("group relative rounded-lg border bg-card cursor-pointer transition-all hover:bg-accent/20 flex flex-col",
        selected ? "border-primary/60 bg-primary/5" : "border-border hover:border-border/80"
      )}>
      <div className="h-28 rounded-t-lg flex items-center justify-center overflow-hidden bg-muted/30 border-b border-border">
        {resource.thumbnail
          ? <img src={resource.thumbnail} alt="" className="h-full w-full object-cover" />
          : resource.url
            ? <Globe className="h-9 w-9 text-muted-foreground/40" />
            : <div className="flex flex-col items-center gap-1 text-muted-foreground/40">
                <FileTypeIcon name={resource.name} type={resource.type} />
                <span className="text-[9px] font-mono uppercase">{(resource.name||"").split(".").pop()}</span>
              </div>
        }
      </div>
      <div className="p-3 flex-1">
        <p className="text-xs font-medium truncate mb-1.5 text-foreground">{resource.name}</p>
        <div className="flex items-center gap-1.5">
          <Badge className={cn("text-[10px] px-1.5 py-0 border", cat.bg, cat.color, cat.border)}>{cat.label}</Badge>
          <span className="text-[10px] text-muted-foreground">{fmtSize(resource.size)}</span>
        </div>
      </div>
      <button onClick={e=>{e.stopPropagation();onDelete(resource.id);}}
        className="absolute top-2 right-2 h-6 w-6 rounded-md bg-background/80 border border-border flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/20 hover:border-destructive/40">
        <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
      </button>
    </div>
  );
}

function ListRow({ resource, onDelete, selected, onSelect }) {
  const cat = CAT[resource.category] || CAT.files;
  return (
    <div onClick={() => onSelect(resource)}
      className={cn("flex items-center gap-3 px-4 py-2.5 rounded-md cursor-pointer transition-all group",
        selected ? "bg-primary/10" : "hover:bg-accent/30"
      )}>
      <div className={cn("h-8 w-8 rounded-md flex items-center justify-center shrink-0", cat.bg)}>
        <FileTypeIcon name={resource.name} type={resource.type} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm truncate text-foreground">{resource.name}</p>
        {resource.url && <p className="text-xs truncate text-muted-foreground">{resource.url}</p>}
      </div>
      <Badge className={cn("shrink-0 text-[10px] border hidden sm:flex", cat.bg, cat.color, cat.border)}>{cat.label}</Badge>
      <span className="text-xs text-muted-foreground w-14 text-right hidden md:block">{fmtSize(resource.size)}</span>
      <span className="text-xs text-muted-foreground w-24 text-right hidden lg:block">{fmtDate(resource.uploadDate)}</span>
      <button onClick={e=>{e.stopPropagation();onDelete(resource.id);}}
        className="h-7 w-7 rounded-md border border-transparent flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:border-destructive/30 hover:bg-destructive/10 shrink-0">
        <Trash2 className="h-3 w-3 text-muted-foreground" />
      </button>
    </div>
  );
}

function PreviewPanel({ resource, onClose }) {
  const cat = CAT[resource.category] || CAT.files;
  const [analysis, setAnalysis] = useState(null);
  const [busy,     setBusy]     = useState(false);

  // Load cached analysis
  useEffect(() => {
    const cache = loadStorage(ANALYSIS_KEY, {});
    if (cache[resource.id]) setAnalysis(cache[resource.id]);
  }, [resource.id]);

  const runAnalysis = async () => {
    if (busy || !resource.name) return;
    setBusy(true);
    const title    = resource.name.replace(/\.[^.]+$/, "");
    const abstract = resource.description || resource.url || title;
    const result   = await analysePaper(title, abstract);
    if (result) {
      setAnalysis(result);
      const cache = loadStorage(ANALYSIS_KEY, {});
      cache[resource.id] = result;
      saveStorage(ANALYSIS_KEY, cache);
    }
    setBusy(false);
  };

  return (
    <div className="w-64 shrink-0 flex flex-col border-l border-border bg-card overflow-hidden animate-slide-in-right">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
        <span className="text-xs font-semibold">Document Details</span>
        <button onClick={onClose} className="h-6 w-6 rounded flex items-center justify-center hover:bg-accent">
          <X className="h-3.5 w-3.5 text-muted-foreground" />
        </button>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Thumbnail */}
          <div className="h-32 rounded-lg bg-muted/30 border border-border flex items-center justify-center overflow-hidden">
            {resource.thumbnail
              ? <img src={resource.thumbnail} alt="" className="h-full w-full object-cover" />
              : resource.url ? <Globe className="h-10 w-10 text-muted-foreground/40" />
              : <FileTypeIcon name={resource.name} type={resource.type} />
            }
          </div>

          {/* Name + link */}
          <div>
            <p className="text-sm font-semibold mb-1 break-words text-foreground">{resource.name}</p>
            {resource.url && (
              <a href={resource.url} target="_blank" rel="noreferrer"
                className="text-xs text-primary flex items-center gap-1 hover:underline">
                <ExternalLink className="h-3 w-3" /> Open source
              </a>
            )}
          </div>

          {/* Metadata */}
          <div className="divide-y divide-border">
            {[
              { label:"Type",    value: cat.label              },
              { label:"Size",    value: fmtSize(resource.size) },
              { label:"Added",   value: fmtDate(resource.uploadDate) },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between py-2">
                <span className="text-xs text-muted-foreground">{label}</span>
                <span className="text-xs font-medium text-foreground">{value}</span>
              </div>
            ))}
          </div>

          <Separator />

          {/* ML Analysis */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-primary" />
                <span className="text-xs font-semibold">ML Analysis</span>
              </div>
              {!analysis && (
                <Button size="sm" variant="outline" className="h-6 text-[10px] px-2" onClick={runAnalysis} disabled={busy}>
                  {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : "Analyse"}
                </Button>
              )}
            </div>

            {!analysis && !busy && (
              <p className="text-xs text-muted-foreground">Click Analyse to classify this document using the ML model.</p>
            )}

            {busy && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Classifying…
              </div>
            )}

            {analysis && (
              <div className="space-y-2 animate-fade-in">
                <div className="rounded-md bg-muted/30 divide-y divide-border">
                  <div className="flex items-center justify-between px-3 py-2">
                    <span className="text-[10px] text-muted-foreground">Category</span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-medium text-foreground">{analysis.predicted_category}</span>
                      <Badge variant="default" className="text-[10px] px-1 py-0">{(analysis.category_confidence*100).toFixed(0)}%</Badge>
                    </div>
                  </div>
                  <div className="flex items-center justify-between px-3 py-2">
                    <span className="text-[10px] text-muted-foreground">Impact</span>
                    <span className={cn("text-xs font-bold", analysis.impact_score > 0.5 ? "text-emerald-400" : "text-amber-400")}>
                      {(analysis.impact_score*100).toFixed(0)}/100
                    </span>
                  </div>
                  <div className="flex items-center justify-between px-3 py-2">
                    <span className="text-[10px] text-muted-foreground">Quality</span>
                    <Badge variant={analysis.quality_label === "well-structured" ? "success" : "destructive"} className="text-[10px]">
                      {analysis.quality_label}
                    </Badge>
                  </div>
                </div>
                <div className="space-y-1.5">
                  <p className="text-[10px] font-medium text-muted-foreground">Top predictions</p>
                  {analysis.top3_categories?.map((c, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground w-16 truncate">{c.category}</span>
                      <div className="flex-1 h-1 rounded-full bg-muted overflow-hidden">
                        <div className="h-full rounded-full bg-primary" style={{width:`${c.confidence*100}%`}}/>
                      </div>
                      <span className="text-[10px] text-muted-foreground w-6 text-right">{(c.confidence*100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function DocumentLibrary() {
  const [resources,  setResources]  = useState(loadResources);
  const [search,     setSearch]     = useState("");
  const [view,       setView]       = useState("grid");
  const [filterCat,  setFilterCat]  = useState("all");
  const [sortBy,     setSortBy]     = useState("date");
  const [selected,   setSelected]   = useState(null);
  const [uploading,  setUploading]  = useState(false);
  const [progress,   setProgress]   = useState(0);
  const [urlInput,   setUrlInput]   = useState("");
  const [tab,        setTab]        = useState("files");
  const [mainView,   setMainView]   = useState("library"); // "library" | "search"

  useEffect(() => { saveResources(resources); }, [resources]);

  // Real file processing — no fake progress timer.
  // Files are read from disk immediately; thumbnails generated for images.
  const simulateUpload = useCallback((files) => {
    setUploading(true); setProgress(0);
    const newResources = files.map(f => ({
      id:          `${Date.now()}-${Math.random()}`,
      name:        f.name,
      size:        f.size,
      type:        f.type,
      category:    f.type?.startsWith("image/") ? "files"
                 : ["application/pdf"].includes(f.type) ? "knowledge"
                 : f.name.endsWith(".md") || f.name.endsWith(".txt") ? "knowledge"
                 : "files",
      uploadDate:  new Date().toISOString(),
      thumbnail:   f.type?.startsWith("image/") ? URL.createObjectURL(f) : null,
    }));
    setResources(prev => [...prev, ...newResources]);
    setProgress(100);
    setUploading(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: simulateUpload, multiple: true, noClick: tab !== "files",
  });

  const addURL = () => {
    if (!urlInput.trim()) return;
    try { new URL(urlInput); } catch { return; }
    setResources(prev => [...prev, {
      id: `${Date.now()}-${Math.random()}`,
      name: urlInput.replace(/^https?:\/\//,"").split("/")[0],
      url: urlInput, category: "webpages",
      uploadDate: new Date().toISOString(), thumbnail: null,
    }]);
    setUrlInput("");
  };

  const importArxiv = (paper) => {
    const exists = resources.some(r => r.url === paper.url);
    if (exists) return;
    setResources(prev => [...prev, {
      id: `arxiv-${paper.id}`,
      name: paper.title,
      url:  paper.url,
      pdfUrl: paper.pdfUrl,
      authors: paper.authors,
      category: "arxiv",
      uploadDate: new Date().toISOString(),
      description: paper.abstract,
      thumbnail: null,
    }]);
  };

  const deleteResource = useCallback((id) => {
    setResources(prev => prev.filter(r => r.id !== id));
    if (selected?.id === id) setSelected(null);
  }, [selected]);

  const filtered = resources
    .filter(r => r.name.toLowerCase().includes(search.toLowerCase()) && (filterCat === "all" || r.category === filterCat))
    .sort((a,b) => sortBy==="date" ? new Date(b.uploadDate)-new Date(a.uploadDate) : sortBy==="name" ? a.name.localeCompare(b.name) : (b.size||0)-(a.size||0));

  const counts = {
    all:      resources.length,
    knowledge:resources.filter(r=>r.category==="knowledge").length,
    files:    resources.filter(r=>r.category==="files").length,
    webpages: resources.filter(r=>r.category==="webpages").length,
    arxiv:    resources.filter(r=>r.category==="arxiv").length,
  };

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-border shrink-0">
          <PageHeader
            title="Document Library"
            description={`${resources.length} documents · ${counts.knowledge} papers · ${counts.arxiv} arXiv · ${counts.webpages} web`}
            actions={
              resources.length > 0 && (
                <Button variant="destructive" size="sm" className="text-xs" onClick={() => { setResources([]); setSelected(null); }}>
                  Clear All
                </Button>
              )
            }
          />
          <div className="flex gap-2 flex-wrap mt-4">
            {[
              { key:"all",       label:"All",       count: counts.all       },
              { key:"arxiv",     label:"arXiv",     count: counts.arxiv     },
              { key:"knowledge", label:"Knowledge", count: counts.knowledge  },
              { key:"files",     label:"Files",     count: counts.files      },
              { key:"webpages",  label:"Web Pages", count: counts.webpages   },
            ].map(({ key, label, count }) => (
              <button key={key} onClick={() => setFilterCat(key)}
                className={cn("inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border transition-all",
                  filterCat === key
                    ? "bg-primary text-primary-foreground border-primary/50"
                    : "bg-transparent border-border text-muted-foreground hover:border-border/80 hover:text-foreground"
                )}>
                {label}
                <span className={cn("rounded-full px-1.5 text-[10px]", filterCat===key?"bg-white/20":"bg-muted")}>{count}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main view — state-controlled to avoid Radix flex height issues */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* View switcher tab bar */}
          <div className="px-6 pt-3 border-b border-border shrink-0">
            <div className="inline-flex h-8 items-center gap-1 rounded-md bg-muted p-1">
              {[
                { id:"library", label:"My Library" },
                { id:"search",  label:"arXiv Search", icon: <Search className="h-3 w-3 mr-1" /> },
              ].map(({ id, label, icon }) => (
                <button key={id} onClick={() => setMainView(id)}
                  className={cn("inline-flex items-center h-6 px-3 rounded text-xs font-medium transition-all",
                    mainView === id
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}>
                  {icon}{label}
                </button>
              ))}
            </div>
          </div>

          {/* ── Library or Search view ── */}
          {mainView === "library" ? (
            <div className="flex-1 overflow-hidden flex flex-col mt-0">
              {/* Toolbar */}
              <div className="flex items-center gap-3 px-6 py-3 border-b border-border shrink-0 flex-wrap">
                <div className="relative flex-1 min-w-40">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                  <Input type="search" placeholder="Search documents…" value={search} onChange={e=>setSearch(e.target.value)} className="pl-8 h-8 text-xs" />
                </div>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-28 h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="date">Recent</SelectItem>
                    <SelectItem value="name">Name</SelectItem>
                    <SelectItem value="size">Size</SelectItem>
                  </SelectContent>
                </Select>
                <div className="flex rounded-md border border-border p-0.5 bg-muted/30">
                  {[["grid",<Grid3X3 className="h-3.5 w-3.5"/>],["list",<List className="h-3.5 w-3.5"/>]].map(([m,icon])=>(
                    <button key={m} onClick={()=>setView(m)}
                      className={cn("flex items-center justify-center h-6 w-6 rounded transition-all",
                        view===m?"bg-background text-foreground shadow-sm":"text-muted-foreground hover:text-foreground"
                      )}>{icon}</button>
                  ))}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto">
                {/* Upload zone */}
                <div className="px-6 pt-4 pb-3 border-b border-border">
                  <div className="flex gap-1 mb-3 p-1 rounded-md bg-muted/30 w-fit border border-border">
                    {[["files","Upload Files"],["url","Add URL"]].map(([t,l])=>(
                      <button key={t} onClick={()=>setTab(t)}
                        className={cn("px-3 py-1 rounded text-xs font-medium transition-all",
                          tab===t?"bg-background text-foreground shadow-sm border border-border":"text-muted-foreground hover:text-foreground"
                        )}>{l}</button>
                    ))}
                  </div>

                  {tab === "files" ? (
                    <div {...getRootProps()}
                      className={cn("rounded-lg border-2 border-dashed px-6 py-6 text-center cursor-pointer transition-all",
                        isDragActive?"border-primary bg-primary/5":"border-border hover:border-border/80 hover:bg-accent/10"
                      )}>
                      <input {...getInputProps()} />
                      {uploading ? (
                        <div className="space-y-2">
                          <Loader2 className="h-6 w-6 mx-auto animate-spin text-primary" />
                          <p className="text-xs font-medium">Uploading…</p>
                          <div className="max-w-xs mx-auto"><Progress value={progress} className="h-1"/></div>
                        </div>
                      ) : (
                        <div>
                          <Upload className={cn("h-6 w-6 mx-auto mb-2", isDragActive?"text-primary":"text-muted-foreground/50")}/>
                          <p className="text-sm font-medium mb-0.5">{isDragActive?"Drop to upload":"Drag & drop files"}</p>
                          <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, MD — or <span className="text-primary">click to browse</span></p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex gap-2">
                      <Input type="url" placeholder="https://arxiv.org/abs/..." value={urlInput}
                        onChange={e=>setUrlInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&addURL()}
                        className="h-9 text-sm"/>
                      <Button onClick={addURL} className="h-9 px-4 shrink-0 gap-1.5">
                        <Link className="h-3.5 w-3.5"/> Add
                      </Button>
                    </div>
                  )}
                </div>

                {/* Documents */}
                <div className="p-6">
                  {filtered.length === 0 ? (
                    <EmptyState
                      icon={FolderOpen}
                      title={resources.length===0?"Library is empty":"No results"}
                      description={resources.length===0?"Upload documents, add URLs, or search arXiv above":"Try a different search or filter"}
                    />
                  ) : view === "grid" ? (
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3">
                      {filtered.map(r=><GridCard key={r.id} resource={r} onDelete={deleteResource} selected={selected?.id===r.id} onSelect={setSelected}/>)}
                    </div>
                  ) : (
                    <div>
                      <div className="flex items-center gap-3 px-4 py-1.5 mb-1">
                        <span className="w-8 shrink-0"/><span className="flex-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Name</span>
                        <span className="hidden sm:block w-20 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Type</span>
                        <span className="hidden md:block w-14 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Size</span>
                        <span className="hidden lg:block w-24 text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Date</span>
                        <span className="w-7 shrink-0"/>
                      </div>
                      <div className="space-y-0.5">
                        {filtered.map(r=><ListRow key={r.id} resource={r} onDelete={deleteResource} selected={selected?.id===r.id} onSelect={setSelected}/>)}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>) : (
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-3xl">
                <div className="mb-5">
                  <h2 className="text-base font-semibold mb-1">Search arXiv</h2>
                  <p className="text-sm text-muted-foreground">
                    2M+ real papers, sorted newest first. Add to your library or run ML classification instantly.
                  </p>
                </div>
                <ArxivSearch onImport={importArxiv} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Preview pane */}
      {selected && <PreviewPanel resource={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

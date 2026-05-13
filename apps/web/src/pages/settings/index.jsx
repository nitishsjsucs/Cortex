import React, { useState, useEffect } from "react";
import {
  User, Brain, Zap, Link2, Bell, Shield,
  Eye, EyeOff, Check, Save, Download, Trash2, ExternalLink,
} from "lucide-react";
import {
  Button, Card, CardContent, CardHeader, CardTitle, CardDescription,
  Input, Label, Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
  Switch, Separator, Badge, Tabs, TabsList, TabsTrigger, TabsContent,
  PageHeader, cn,
} from "../../components/ui/index.jsx";

const SETTINGS_KEY = "cortex_settings_v2";
const DEFAULT = {
  profile: { name: "Nitish Chowdary", email: "nitish@research.edu", org: "Research Lab", role: "ML Engineer" },
  ai:      { provider: "openai", apiKey: "", model: "gpt-4o", inferenceEndpoint: "http://localhost:8001", mlflowEndpoint: "http://localhost:5000" },
  research:{ defaultDepth: "comprehensive", citationStyle: "APA", autoSave: true, showSources: true },
  notifications: { researchComplete: true, weeklyDigest: false, systemUpdates: true },
  privacy: { shareAnalytics: false, dataRetention: "90" },
};

const load = () => { try { return { ...DEFAULT, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; } catch { return DEFAULT; } };

function SettingRow({ label, description, children }) {
  return (
    <div className="flex items-center justify-between py-4 border-b border-border last:border-0">
      <div className="flex-1 min-w-0 mr-8">
        <p className="text-sm font-medium">{label}</p>
        {description && <p className="text-xs text-muted-foreground mt-0.5">{description}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function SectionCard({ title, description, children }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export default function Settings() {
  const [s,       setS]       = useState(load);
  const [saved,   setSaved]   = useState(false);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch {}
  }, [s]);

  const set = (section, key, value) => setS(p => ({ ...p, [section]: { ...p[section], [key]: value } }));

  const save = () => { setSaved(true); setTimeout(() => setSaved(false), 2000); };

  const TABS = [
    { id: "profile",       label: "Profile",       icon: User   },
    { id: "ai",            label: "AI & Model",    icon: Brain  },
    { id: "research",      label: "Research",      icon: Zap    },
    { id: "integrations",  label: "Integrations",  icon: Link2  },
    { id: "notifications", label: "Notifications", icon: Bell   },
    { id: "privacy",       label: "Privacy",       icon: Shield },
  ];

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <PageHeader
        title="Settings"
        description="Manage your account, AI model, and preferences"
        actions={
          <Button onClick={save} size="sm" variant={saved ? "outline" : "default"} className={cn(saved && "text-emerald-400 border-emerald-400/40")}>
            {saved ? <><Check className="h-3.5 w-3.5" /> Saved!</> : <><Save className="h-3.5 w-3.5" /> Save Changes</>}
          </Button>
        }
      />

      <Tabs defaultValue="profile">
        <TabsList className="h-auto flex-wrap gap-1 bg-transparent p-0 justify-start">
          {TABS.map(({ id, label, icon: Icon }) => (
            <TabsTrigger key={id} value={id}
              className="h-8 px-3 text-xs data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md border border-transparent data-[state=active]:border-primary/30">
              <Icon className="h-3.5 w-3.5 mr-1.5" />
              {label}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* Profile */}
        <TabsContent value="profile">
          <SectionCard title="Profile Information" description="Your personal details and display name">
            <SettingRow label="Full Name" description="Displayed across the platform">
              <Input value={s.profile.name} onChange={e => set("profile","name",e.target.value)} className="w-52" />
            </SettingRow>
            <SettingRow label="Email" description="Primary contact email">
              <Input type="email" value={s.profile.email} onChange={e => set("profile","email",e.target.value)} className="w-52" />
            </SettingRow>
            <SettingRow label="Organisation">
              <Input value={s.profile.org} onChange={e => set("profile","org",e.target.value)} className="w-52" />
            </SettingRow>
            <SettingRow label="Role">
              <Input value={s.profile.role} onChange={e => set("profile","role",e.target.value)} className="w-52" />
            </SettingRow>
          </SectionCard>
        </TabsContent>

        {/* AI & Model */}
        <TabsContent value="ai">
          <SectionCard title="AI Configuration" description="LLM provider and inference endpoints">
            <SettingRow label="AI Provider" description="LLM for research generation">
              <Select value={s.ai.provider} onValueChange={v => set("ai","provider",v)}>
                <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="local">Local (Ollama)</SelectItem>
                </SelectContent>
              </Select>
            </SettingRow>
            <SettingRow label="Model">
              <Select value={s.ai.model} onValueChange={v => set("ai","model",v)}>
                <SelectTrigger className="w-44"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                  <SelectItem value="gpt-4o-mini">GPT-4o-mini</SelectItem>
                  <SelectItem value="claude-sonnet-4-6">Claude Sonnet 4.6</SelectItem>
                  <SelectItem value="llama3.2">Llama 3.2 (local)</SelectItem>
                </SelectContent>
              </Select>
            </SettingRow>
            <SettingRow label="API Key" description="Stored locally only">
              <div className="flex items-center gap-1.5">
                <Input
                  type={showKey ? "text" : "password"}
                  value={s.ai.apiKey}
                  onChange={e => set("ai","apiKey",e.target.value)}
                  placeholder="sk-…"
                  className="w-44 font-mono text-xs"
                />
                <Button variant="ghost" size="icon" onClick={() => setShowKey(p => !p)} className="h-9 w-9 shrink-0">
                  {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                </Button>
              </div>
            </SettingRow>
            <SettingRow label="Inference API" description="Local ML model endpoint">
              <div className="flex items-center gap-1.5">
                <Input value={s.ai.inferenceEndpoint} onChange={e => set("ai","inferenceEndpoint",e.target.value)} className="w-44 font-mono text-xs" />
                <a href={s.ai.inferenceEndpoint + "/docs"} target="_blank" rel="noreferrer">
                  <Button variant="ghost" size="icon" className="h-9 w-9 shrink-0"><ExternalLink className="h-3.5 w-3.5" /></Button>
                </a>
              </div>
            </SettingRow>
            <SettingRow label="MLflow" description="Experiment tracking">
              <div className="flex items-center gap-1.5">
                <Input value={s.ai.mlflowEndpoint} onChange={e => set("ai","mlflowEndpoint",e.target.value)} className="w-44 font-mono text-xs" />
                <a href={s.ai.mlflowEndpoint} target="_blank" rel="noreferrer">
                  <Button variant="ghost" size="icon" className="h-9 w-9 shrink-0"><ExternalLink className="h-3.5 w-3.5" /></Button>
                </a>
              </div>
            </SettingRow>
          </SectionCard>
        </TabsContent>

        {/* Research */}
        <TabsContent value="research">
          <SectionCard title="Research Preferences" description="Default settings for research workflows">
            <SettingRow label="Default Research Depth" description="How many sources per project">
              <Select value={s.research.defaultDepth} onValueChange={v => set("research","defaultDepth",v)}>
                <SelectTrigger className="w-48"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="quick">Quick (5 sources)</SelectItem>
                  <SelectItem value="comprehensive">Comprehensive (15)</SelectItem>
                  <SelectItem value="exhaustive">Exhaustive (30+)</SelectItem>
                </SelectContent>
              </Select>
            </SettingRow>
            <SettingRow label="Citation Style">
              <Select value={s.research.citationStyle} onValueChange={v => set("research","citationStyle",v)}>
                <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["APA","MLA","Chicago","IEEE","Harvard"].map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </SettingRow>
            <SettingRow label="Auto-save" description="Save state every 30 seconds">
              <Switch checked={s.research.autoSave} onCheckedChange={v => set("research","autoSave",v)} />
            </SettingRow>
            <SettingRow label="Show Source Details" description="Display relevance scores">
              <Switch checked={s.research.showSources} onCheckedChange={v => set("research","showSources",v)} />
            </SettingRow>
          </SectionCard>
        </TabsContent>

        {/* Integrations */}
        <TabsContent value="integrations">
          <SectionCard title="External Integrations" description="Connect third-party research services">
            <div className="space-y-3">
              {[
                { name: "arXiv API",        desc: "Pre-print paper database",             connected: true,  color: "text-emerald-400" },
                { name: "Semantic Scholar",  desc: "Academic search & citation data",      connected: false, color: "text-indigo-400"  },
                { name: "Zotero",            desc: "Reference management",                 connected: false, color: "text-amber-400"   },
                { name: "Google Scholar",    desc: "Web-based academic search",            connected: false, color: "text-cyan-400"    },
              ].map(({ name, desc, connected, color }) => (
                <div key={name} className="flex items-center gap-4 rounded-lg border border-border bg-muted/20 px-4 py-3">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">{name}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{desc}</p>
                  </div>
                  <Badge variant={connected ? "success" : "secondary"}>
                    {connected ? "Connected" : "Not connected"}
                  </Badge>
                  <Button variant={connected ? "destructive" : "outline"} size="sm" className="h-7 text-xs">
                    {connected ? "Disconnect" : "Connect"}
                  </Button>
                </div>
              ))}
            </div>
          </SectionCard>
        </TabsContent>

        {/* Notifications */}
        <TabsContent value="notifications">
          <SectionCard title="Notifications" description="Control when and how you're notified">
            <SettingRow label="Research Complete" description="When a workflow finishes">
              <Switch checked={s.notifications.researchComplete} onCheckedChange={v => set("notifications","researchComplete",v)} />
            </SettingRow>
            <SettingRow label="Weekly Digest" description="Summary of research activity">
              <Switch checked={s.notifications.weeklyDigest} onCheckedChange={v => set("notifications","weeklyDigest",v)} />
            </SettingRow>
            <SettingRow label="System Updates" description="New features and updates">
              <Switch checked={s.notifications.systemUpdates} onCheckedChange={v => set("notifications","systemUpdates",v)} />
            </SettingRow>
          </SectionCard>
        </TabsContent>

        {/* Privacy */}
        <TabsContent value="privacy">
          <SectionCard title="Privacy & Data" description="Control your data and privacy settings">
            <SettingRow label="Share Analytics" description="Anonymous usage data to improve Cortex">
              <Switch checked={s.privacy.shareAnalytics} onCheckedChange={v => set("privacy","shareAnalytics",v)} />
            </SettingRow>
            <SettingRow label="Data Retention" description="How long to keep research history">
              <Select value={s.privacy.dataRetention} onValueChange={v => set("privacy","dataRetention",v)}>
                <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {["30","60","90","180","365"].map(d => <SelectItem key={d} value={d}>{d} days</SelectItem>)}
                </SelectContent>
              </Select>
            </SettingRow>
            <div className="pt-4 flex flex-wrap gap-3">
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="h-3.5 w-3.5" /> Export all data
              </Button>
              <Button variant="destructive" size="sm" className="gap-2">
                <Trash2 className="h-3.5 w-3.5" /> Delete history
              </Button>
            </div>
          </SectionCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}

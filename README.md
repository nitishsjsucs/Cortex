<h1 align="center">Cortex</h1>

<p align="center">
  <strong>Enterprise AI Research Platform</strong><br>
  <em>CQRS/Event Sourcing &middot; Multi-Agent Orchestration &middot; RAG &middot; MLOps &middot; Real-Time Analytics</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/version-5.0.0-blue?style=flat-square" alt="v5.0.0">
  <img src="https://img.shields.io/badge/Rust-Actix--web-orange?style=flat-square&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/React_18-TypeScript-61dafb?style=flat-square&logo=react" alt="React 18">
  <img src="https://img.shields.io/badge/Kubernetes-native-326ce5?style=flat-square&logo=kubernetes&logoColor=white" alt="K8s">
  <img src="https://img.shields.io/badge/GraphQL-API-e10098?style=flat-square&logo=graphql" alt="GraphQL">
  <img src="https://img.shields.io/badge/MLOps-Kubeflow-blue?style=flat-square" alt="MLOps">
  <img src="https://img.shields.io/badge/SOC2%20|%20GDPR%20|%20HIPAA-compliant-red?style=flat-square" alt="Compliance">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#core-systems">Core Systems</a> &middot;
  <a href="#ai-capabilities">AI Capabilities</a> &middot;
  <a href="#deployment">Deployment</a> &middot;
  <a href="#technology-stack">Stack</a>
</p>

---

**Cortex** is a production-grade AI research platform built on **event-sourced CQRS architecture** in Rust, with a React frontend, GraphQL API gateway, and cloud-native Kubernetes infrastructure. It orchestrates multi-agent research workflows — from query to synthesized findings — using RAG pipelines, vector search, and multi-provider LLM integration.

The platform implements 9 enterprise infrastructure phases: event sourcing, CQRS, Kubernetes + Istio service mesh, GraphQL with real-time subscriptions, serverless edge computing, MLOps (Kubeflow + MLflow + TF Serving), real-time analytics (ClickHouse + Kafka + Airflow), multi-tenant enterprise auth (Keycloak), and zero-trust security with SOC 2 / GDPR / HIPAA compliance.

**50,000+ lines of production code** &middot; **<100ms P95 inference latency** &middot; **50,000+ concurrent users** &middot; **99.9% uptime target**

## Quick Start

```bash
# Production deployment
git clone https://github.com/cortex-research/cortex.git
cd cortex
./scripts/production-startup.sh

# Local development
cd apps/web && npm run dev        # React frontend
cd apps/desktop && npm run dev    # Tauri desktop app
```

**Prerequisites:** Kubernetes v1.28+, `kubectl`, `helm`, `istioctl`, `docker`

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Cortex v5.0                                     │
│                     Enterprise AI Research Platform                      │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 1  │  Event Sourcing + CQRS (Rust, PostgreSQL, Aggregates)       │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 2  │  Kubernetes + Istio Service Mesh (HA, multi-zone)           │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 3  │  GraphQL API Gateway + Real-time WebSocket Subscriptions    │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 4  │  Knative Serverless + Edge Computing                        │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 5  │  MLOps (Kubeflow + MLflow + TensorFlow Serving + GPU)       │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 6  │  Analytics (ClickHouse + Kafka + Airflow)                   │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 7  │  Multi-tenant Enterprise (Keycloak + RBAC + Billing)        │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 8  │  Security & Compliance (Vault + Velero + Falco)             │
├───────────┼──────────────────────────────────────────────────────────────┤
│  Layer 9  │  AI Enhancement (RAG + Qdrant + Ollama + MCP)               │
└───────────┴──────────────────────────────────────────────────────────────┘
```

## Core Systems

### Event Sourcing + CQRS (Rust)

The backbone of Cortex is a **Rust-based event-sourced CQRS** system with domain-driven design:

- **Aggregate Root pattern** for research workflow state management with optimistic concurrency
- **`AggregatePersistence` trait** — shared load-mutate-save pipeline eliminates duplication across all command handlers
- **`impl_correlation_id!` macro** — generates correlation ID extraction for event enums, keeping the pattern DRY as new variants are added
- **Domain events** with schema versioning, correlation/causation tracking, and validation
- **PostgreSQL event store** with append-only streams and snapshot support
- **Read model projections** for optimized query-side views

### GraphQL API Gateway

Unified API with real-time capabilities:

- **Queries & Mutations** — research workflows, API key management, federated research, BMAD agent orchestration
- **DataLoader pattern** — N+1 query elimination via batched data loading
- **Real-time subscriptions** — WebSocket-based live updates for workflow progress
- **Auth middleware** — per-resolver authentication with admin-level gating
- **Rate limiting** — per-user and per-API-key throttling

### Multi-Agent Orchestration

The `ai-orchestrator` package coordinates multiple specialized AI agents:

| Agent | Role | Research Types |
|-------|------|----------------|
| **Researcher** | Primary investigation | Market, technology, competitive |
| **Architect** | System design analysis | Architecture patterns, infrastructure |
| **Security** | Compliance research | Security, compliance, privacy |
| **Analyst** | Data-driven insights | User research, analytics |

Agents execute workflows via the `WorkflowCoordinationService` with quality gates, cost tracking, and parallel phase execution.

## AI Capabilities

### RAG Pipeline (Phase 5.0)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Qdrant v1.11.0 | High-performance similarity search |
| **Embeddings** | OpenAI / Hugging Face | Semantic document embedding |
| **Local LLM** | Ollama + GPU | Privacy-preserving local inference |
| **MCP Server** | Model Context Protocol | Standardized AI model communication |
| **Cost Router** | Hybrid optimizer | Intelligent model routing by cost/quality |

### MLOps Pipeline

- **Kubeflow Pipelines** — automated ML workflow orchestration with DAG-based steps
- **MLflow Model Registry** — versioning, metadata, experiment tracking, A/B testing
- **TensorFlow Serving** — GPU-accelerated inference at **<100ms P95 latency**
- **Model monitoring** — drift detection, performance degradation alerts

### Multi-Provider LLM Support

OpenAI, Hugging Face, Groq, Together AI, Replicate, Ollama (local) — with cost optimization and automatic fallback routing.

## Analytics & Observability

| Technology | Function | Scale |
|-----------|----------|-------|
| **ClickHouse** | Columnar data warehouse | Petabyte-scale, <1hr pipeline latency |
| **Apache Kafka** | Streaming data processing | Real-time event ingestion |
| **Apache Airflow** | ETL orchestration | Automated data pipelines |
| **Grafana** | Dashboards & BI | Self-service reporting |
| **Prometheus** | Metrics collection | System-wide observability |
| **Jaeger** | Distributed tracing | Request-level debugging |

## Enterprise Features

### Multi-Tenant Architecture

- **Namespace isolation** — Kubernetes-based tenant separation
- **Keycloak SSO** — SAML, OAuth2, MFA enterprise authentication
- **RBAC** — granular role-based access control
- **Billing engine** — automated usage tracking and resource metering
- **White-label** — custom branding and domain configuration

### Security & Compliance

- **Zero-trust architecture** — mTLS everywhere, network policies, Falco runtime protection
- **HashiCorp Vault** — centralized secrets management and rotation
- **Velero** — disaster recovery with 4-hour RTO, 1-hour RPO
- **Compliance** — SOC 2, GDPR, HIPAA certified frameworks

## Repository Structure

```
cortex/
├── apps/
│   ├── desktop/                    # Tauri cross-platform desktop app
│   └── web/                        # React 18 + Tailwind frontend
├── packages/
│   ├── ai-orchestrator/            # Rust CQRS/ES backend + GraphQL
│   │   ├── core/                   # Event store, aggregates, CQRS handlers
│   │   ├── graphql/                # GraphQL schema, resolvers, mutations
│   │   └── integration/            # BMAD agent bridge, workflow coordinator
│   ├── bmad-core/                  # BMAD agent configurations
│   ├── rag-engine/                 # RAG pipeline + vector search
│   ├── mcp-server/                 # Model Context Protocol server
│   └── serverless-functions/       # Knative serverless functions
├── infrastructure/
│   └── kubernetes/                 # K8s manifests for all 9 phases
├── docker/                         # Docker configs (dev + prod)
├── docs/                           # Architecture, API, deployment guides
└── scripts/                        # Production deployment automation
```

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript, Tailwind, Vite, Tauri, Redux Toolkit, D3.js |
| **Backend** | Rust (Actix-web), Event Sourcing, CQRS, Domain Events |
| **API** | GraphQL (async-graphql), WebSocket subscriptions, DataLoader |
| **Database** | PostgreSQL 15 (event store), Redis 7 (cache), ClickHouse (analytics) |
| **AI/ML** | Kubeflow, MLflow, TensorFlow Serving, Qdrant, Ollama, MCP |
| **Streaming** | Apache Kafka, Apache Airflow |
| **Auth** | Keycloak (SAML, OAuth2, MFA), RBAC, JWT |
| **Infra** | Kubernetes, Istio, Knative, Helm |
| **Security** | HashiCorp Vault, Falco, Velero, mTLS |
| **Monitoring** | Prometheus, Grafana, Jaeger |

## Performance

| Metric | Target |
|--------|--------|
| **System uptime** | 99.9% |
| **API P95 latency** | <200ms |
| **ML inference P95** | <100ms |
| **Analytics pipeline** | <1 hour |
| **Concurrent users** | 50,000+ |
| **Horizontal scaling** | Auto (GPU-aware) |

## Deployment

```bash
# Full production deployment
./scripts/production-startup.sh

# Individual infrastructure phases
cd infrastructure/kubernetes
./deploy-phase-4.6.sh   # MLOps pipeline
./deploy-phase-4.7.sh   # Analytics stack
./deploy-phase-4.8.sh   # Enterprise auth + billing
./deploy-phase-4.9.sh   # Security + compliance
./deploy-phase-5.0.sh   # AI enhancement (RAG + LLM)

# Docker
npm run docker:build:prod
npm run docker:up:prod
```

## Documentation

- [Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)
- [Architecture Documentation](docs/architecture/)
- [API Reference](docs/api/README.md)
- [BMAD Agent Guide](docs/user-guides/bmad-agents.md)
- [MLOps Guide](infrastructure/kubernetes/README-PHASE-4.6.md)
- [Security & Compliance](infrastructure/kubernetes/README-PHASE-4.9.md)

## My Contributions

- **Event-Sourced CQRS Backend** — Architected and implemented the core Rust backend with event sourcing, CQRS command/query separation, and domain-driven aggregate design for research document management.
- **Multi-Agent Orchestration** — Built the AI agent orchestration layer with RAG pipelines, dynamic tool selection, and parallel agent execution for complex research workflows.
- **Real-Time Analytics Stack** — Developed the analytics pipeline with streaming event processing, materialized views, and dashboard APIs for research activity monitoring.
- **GraphQL API Layer** — Designed the type-safe GraphQL API with subscription support for real-time updates, batch operations, and fine-grained access control.
- **Kubernetes Infrastructure** — Created the complete Kubernetes deployment infrastructure including Helm charts, horizontal pod autoscaling, service mesh configuration, and CI/CD pipelines.

---

## License

[MIT](LICENSE)

---

<p align="center">
  <sub>50,000+ lines of production Rust, TypeScript, and infrastructure-as-code.</sub>
</p>

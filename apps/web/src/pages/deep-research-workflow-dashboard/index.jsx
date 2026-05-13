import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../../components/ui/Header';
import WorkflowProgressNavigator from '../../components/ui/WorkflowProgressNavigator';
import ResearchTopicsSection from './components/ResearchTopicsSection';
import QuestionSection from './components/QuestionSection';
import InformationCollectionSection from './components/InformationCollectionSection';
import FinalReportSection from './components/FinalReportSection';
import WorkflowControls from './components/WorkflowControls';

// ── localStorage helpers ───────────────────────────────────────────────────

const STORAGE_KEY = 'cortex_workflow_state';

const defaultWorkflowPersist = {
  researchTopics: '',
  workflowState: { isActive: false, progress: 0, currentStage: 0 },
  expandedSections: { topics: true, questions: false, collection: false, report: false },
  questionStatus: 'waiting',
  collectionStatus: 'waiting',
  reportStatus: 'waiting',
  collectionProgress: 0,
};

const loadWorkflowState = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultWorkflowPersist;
    return { ...defaultWorkflowPersist, ...JSON.parse(raw) };
  } catch {
    return defaultWorkflowPersist;
  }
};

const saveWorkflowState = (state) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* ignore quota errors */ }
};

// ── Component ──────────────────────────────────────────────────────────────

const DeepResearchWorkflowDashboard = () => {
  const navigate = useNavigate();

  const initial = loadWorkflowState();

  const [workflowState, setWorkflowState]       = useState(initial.workflowState);
  const [expandedSections, setExpandedSections] = useState(initial.expandedSections);
  const [researchTopics, setResearchTopics]     = useState(initial.researchTopics);
  const [questionStatus, setQuestionStatus]     = useState(initial.questionStatus);
  const [collectionStatus, setCollectionStatus] = useState(initial.collectionStatus);
  const [reportStatus, setReportStatus]         = useState(initial.reportStatus);
  const [collectionProgress, setCollectionProgress] = useState(initial.collectionProgress);

  // Persist all state to localStorage on every change
  useEffect(() => {
    saveWorkflowState({
      researchTopics,
      workflowState,
      expandedSections,
      questionStatus,
      collectionStatus,
      reportStatus,
      collectionProgress,
    });
  }, [researchTopics, workflowState, expandedSections, questionStatus, collectionStatus, reportStatus, collectionProgress]);

  // Mock data (static)
  const questions = [
    "What are the current market trends and growth projections for artificial intelligence in healthcare?",
    "How do regulatory frameworks impact AI implementation in medical diagnostics?",
    "What are the key challenges and barriers to AI adoption in healthcare institutions?",
    "Which AI technologies show the most promise for improving patient outcomes?",
    "How do healthcare professionals perceive AI integration in their workflow?"
  ];

  const collectedSources = [
    { title: "AI in Healthcare: Market Analysis 2024", description: "Comprehensive market research on AI adoption in healthcare sector", type: "web", relevance: 95 },
    { title: "FDA Guidelines for AI Medical Devices", description: "Official regulatory framework for AI-powered medical devices", type: "document", relevance: 88 },
    { title: "Healthcare AI Implementation Case Studies", description: "Real-world examples of successful AI integration in hospitals", type: "web", relevance: 92 },
    { title: "Machine Learning in Diagnostics Research Paper", description: "Peer-reviewed research on ML applications in medical diagnostics", type: "document", relevance: 90 },
    { title: "Healthcare Professional AI Survey Results", description: "Survey data on healthcare worker attitudes toward AI technology", type: "web", relevance: 85 },
  ];

  const reportData = { pages: 24, sources: 15, insights: 8 };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleStartWorkflow = () => {
    if (!researchTopics.trim()) {
      alert('Please define your research topics first');
      return;
    }

    setWorkflowState(prev => ({ ...prev, isActive: true }));

    setTimeout(() => {
      setQuestionStatus('processing');
      setWorkflowState(prev => ({ ...prev, progress: 25, currentStage: 1 }));
    }, 1000);

    setTimeout(() => {
      setQuestionStatus('completed');
      setCollectionStatus('collecting');
      setWorkflowState(prev => ({ ...prev, progress: 50, currentStage: 2 }));

      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        setCollectionProgress(progress);
        if (progress >= 100) {
          clearInterval(interval);
          setCollectionStatus('completed');
          setReportStatus('ready');
          setWorkflowState(prev => ({ ...prev, progress: 75, currentStage: 3 }));
        }
      }, 500);
    }, 3000);
  };

  const handlePauseWorkflow = () => {
    setWorkflowState(prev => ({ ...prev, isActive: false }));
  };

  const handleResetWorkflow = () => {
    setWorkflowState({ isActive: false, progress: 0, currentStage: 0 });
    setQuestionStatus('waiting');
    setCollectionStatus('waiting');
    setReportStatus('waiting');
    setCollectionProgress(0);
    setResearchTopics('');
    setExpandedSections({ topics: true, questions: false, collection: false, report: false });
  };

  const handleAddResource = (type) => {
    if (type === 'file') { navigate('/resource-upload-management'); return; }
    if (type === 'webpage') {
      const url = prompt('Enter webpage URL:');
      if (url) alert(`Webpage "${url}" will be added to resources`);
      return;
    }
    alert('Knowledge base integration coming soon');
  };

  const handleGenerateReport = () => {
    setReportStatus('generating');
    setTimeout(() => {
      setReportStatus('completed');
      setWorkflowState(prev => ({ ...prev, progress: 100, currentStage: 4 }));
    }, 3000);
  };

  const handleViewReport = () => { navigate('/final-report-generation'); };

  // Mark questions "ready" once topic is filled
  useEffect(() => {
    if (researchTopics.trim() && questionStatus === 'waiting' && !workflowState.isActive) {
      setQuestionStatus('ready');
    } else if (!researchTopics.trim() && questionStatus === 'ready') {
      setQuestionStatus('waiting');
    }
  }, [researchTopics]);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="container mx-auto px-6 py-8 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Deep Research Workflow Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Conduct comprehensive research through our structured 4-step process
          </p>
        </div>

        <div className="mb-8">
          <WorkflowProgressNavigator />
        </div>

        <div className="mb-8">
          <WorkflowControls
            isWorkflowActive={workflowState.isActive}
            onStartWorkflow={handleStartWorkflow}
            onPauseWorkflow={handlePauseWorkflow}
            onResetWorkflow={handleResetWorkflow}
            workflowProgress={workflowState.progress}
          />
        </div>

        <div className="space-y-6">
          <ResearchTopicsSection
            isExpanded={expandedSections.topics}
            onToggle={() => toggleSection('topics')}
            researchTopics={researchTopics}
            onTopicsChange={setResearchTopics}
            onAddResource={handleAddResource}
          />

          <QuestionSection
            isExpanded={expandedSections.questions}
            onToggle={() => toggleSection('questions')}
            status={questionStatus}
            questions={questions}
          />

          <InformationCollectionSection
            isExpanded={expandedSections.collection}
            onToggle={() => toggleSection('collection')}
            status={collectionStatus}
            progress={collectionProgress}
            collectedSources={collectedSources}
          />

          <FinalReportSection
            isExpanded={expandedSections.report}
            onToggle={() => toggleSection('report')}
            status={reportStatus}
            reportData={reportData}
            onGenerateReport={handleGenerateReport}
            onViewReport={handleViewReport}
          />
        </div>

        <footer className="mt-16 pt-8 border-t border-border text-center">
          <p className="text-muted-foreground">
            Cortex Research Platform — {new Date().getFullYear()}
          </p>
        </footer>
      </main>
    </div>
  );
};

export default DeepResearchWorkflowDashboard;

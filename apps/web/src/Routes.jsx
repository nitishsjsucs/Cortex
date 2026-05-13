import React from 'react';
import { BrowserRouter, Routes as RouterRoutes, Route, Navigate } from 'react-router-dom';
import ScrollToTop from 'components/ScrollToTop';
import ErrorBoundary from 'components/ErrorBoundary';
import AppLayout from 'components/layout/AppLayout';

import Dashboard    from 'pages/dashboard';
import ResearchWorkspace from 'pages/research-workspace';
import DocumentLibrary   from 'pages/document-library';
import Analytics         from 'pages/analytics';
import ReportEditor      from 'pages/report-editor';
import MLPlatform        from 'pages/ml-platform';
import Settings          from 'pages/settings';
import NotFound          from 'pages/NotFound';

const Routes = () => (
  <BrowserRouter>
    <ErrorBoundary>
      <ScrollToTop />
      <RouterRoutes>
        <Route path="/"          element={<AppLayout><Dashboard /></AppLayout>} />
        <Route path="/workspace" element={<AppLayout><ResearchWorkspace /></AppLayout>} />
        <Route path="/library"   element={<AppLayout><DocumentLibrary /></AppLayout>} />
        <Route path="/platform"  element={<AppLayout><MLPlatform /></AppLayout>} />
        <Route path="/analytics" element={<AppLayout><Analytics /></AppLayout>} />
        <Route path="/report"    element={<AppLayout><ReportEditor /></AppLayout>} />
        <Route path="/settings"  element={<AppLayout><Settings /></AppLayout>} />

        {/* Legacy redirects */}
        <Route path="/deep-research-workflow-dashboard" element={<Navigate to="/workspace" replace />} />
        <Route path="/resource-upload-management"       element={<Navigate to="/library"   replace />} />
        <Route path="/research-progress-tracking"       element={<Navigate to="/analytics" replace />} />
        <Route path="/final-report-generation"          element={<Navigate to="/report"    replace />} />
        <Route path="/settings-and-configuration"       element={<Navigate to="/settings"  replace />} />

        <Route path="*" element={<NotFound />} />
      </RouterRoutes>
    </ErrorBoundary>
  </BrowserRouter>
);

export default Routes;

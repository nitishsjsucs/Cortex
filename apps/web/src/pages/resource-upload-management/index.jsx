import React, { useState, useCallback, useEffect } from 'react';
import Header from '../../components/ui/Header';
import Input from '../../components/ui/Input';
import ResourceTypeSelector from './components/ResourceTypeSelector';
import UploadZone from './components/UploadZone';
import ResourceList from './components/ResourceList';
import WebPageUpload from './components/WebPageUpload';

const STORAGE_KEY = 'cortex_uploaded_resources';

const loadFromStorage = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    // thumbnail ObjectURLs don't survive reload — strip them
    return JSON.parse(raw).map(r => ({ ...r, thumbnail: r.thumbnail?.startsWith('blob:') ? null : r.thumbnail }));
  } catch {
    return [];
  }
};

const saveToStorage = (resources) => {
  try {
    // Don't persist blob URLs (they're session-only)
    const serialisable = resources.map(r => ({ ...r, thumbnail: r.thumbnail?.startsWith('blob:') ? null : r.thumbnail }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serialisable));
  } catch { /* quota exceeded — ignore */ }
};

const ResourceUploadManagement = () => {
  const [selectedResourceType, setSelectedResourceType] = useState('knowledge');
  const [uploadedResources, setUploadedResources] = useState(loadFromStorage);
  const [searchQuery, setSearchQuery] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [filterType, setFilterType] = useState('all');

  // Persist resources to localStorage whenever they change
  useEffect(() => {
    saveToStorage(uploadedResources);
  }, [uploadedResources]);

  // Simulate file upload — fast enough that the UI feels responsive (50ms steps)
  const simulateUpload = useCallback((files) => {
    setIsUploading(true);
    setUploadProgress(0);

    const uploadInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(uploadInterval);
          setIsUploading(false);

          const newResources = files.map(file => ({
            id: `${Date.now()}-${Math.random()}`,
            name: file.name,
            size: file.size,
            type: file.type,
            category: selectedResourceType,
            uploadDate: new Date().toISOString(),
            thumbnail: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
          }));

          setUploadedResources(prev => [...prev, ...newResources]);
          setUploadProgress(0);
          return 0;
        }
        return prev + 20; // 5 steps × 50ms = 250ms total
      });
    }, 50);
  }, [selectedResourceType]);

  const handleFileDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) simulateUpload(acceptedFiles);
  }, [simulateUpload]);

  const handleWebPageAdd = useCallback((urlData) => {
    const newResource = {
      id: `${Date.now()}-${Math.random()}`,
      name: urlData.title || urlData.url,
      url: urlData.url,
      description: urlData.description,
      category: 'webpages',
      uploadDate: new Date().toISOString(),
      thumbnail: urlData.favicon || null,
    };
    setUploadedResources(prev => [...prev, newResource]);
  }, []);

  const handleResourceRemove = useCallback((resourceId) => {
    setUploadedResources(prev => prev.filter(r => r.id !== resourceId));
  }, []);

  const filteredResources = uploadedResources.filter(resource => {
    const matchesSearch = resource.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === 'all' || resource.category === filterType;
    return matchesSearch && matchesType;
  });

  const getAcceptedFileTypes = () => {
    switch (selectedResourceType) {
      case 'knowledge': return '.pdf,.doc,.docx,.txt,.md,.rtf';
      case 'files':     return '.pdf,.doc,.docx,.txt,.md,.rtf,.jpg,.jpeg,.png,.gif,.xlsx,.xls,.ppt,.pptx';
      default:          return '.pdf,.doc,.docx,.txt,.md,.rtf';
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Resource Upload Management
          </h1>
          <p className="text-muted-foreground">
            Add and manage research resources across knowledge, local files, and web pages
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Resource Type Selector */}
          <div className="lg:col-span-3">
            <ResourceTypeSelector
              selectedType={selectedResourceType}
              onTypeChange={setSelectedResourceType}
            />
          </div>

          {/* Upload Zone */}
          <div className="lg:col-span-6">
            <div className="space-y-6">
              {selectedResourceType === 'webpages' ? (
                <WebPageUpload onWebPageAdd={handleWebPageAdd} />
              ) : (
                <UploadZone
                  onFileDrop={handleFileDrop}
                  acceptedTypes={getAcceptedFileTypes()}
                  isUploading={isUploading}
                  uploadProgress={uploadProgress}
                  resourceType={selectedResourceType}
                />
              )}
            </div>
          </div>

          {/* Resource List */}
          <div className="lg:col-span-3">
            <div className="space-y-4">
              <div className="space-y-3">
                <Input
                  type="search"
                  placeholder="Search resources..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                />
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="w-full px-3 py-2 bg-card border border-border rounded-md text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="all">All Types</option>
                  <option value="knowledge">Knowledge</option>
                  <option value="files">Local Files</option>
                  <option value="webpages">Web Pages</option>
                </select>
              </div>

              <ResourceList
                resources={filteredResources}
                onResourceRemove={handleResourceRemove}
              />
            </div>
          </div>
        </div>

        {/* Mobile stats */}
        <div className="lg:hidden mt-8 space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div className="research-card p-4 text-center">
              <div className="text-2xl font-bold text-primary mb-1">{uploadedResources.length}</div>
              <div className="text-sm text-muted-foreground">Total Resources</div>
            </div>
            <div className="research-card p-4 text-center">
              <div className="text-2xl font-bold text-accent mb-1">
                {uploadedResources.filter(r => r.category === selectedResourceType).length}
              </div>
              <div className="text-sm text-muted-foreground">
                {selectedResourceType === 'knowledge' ? 'Knowledge' :
                 selectedResourceType === 'files' ? 'Local Files' : 'Web Pages'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResourceUploadManagement;

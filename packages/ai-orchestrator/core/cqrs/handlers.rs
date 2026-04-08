// Command and Query Handlers for CQRS
// Phase 4.2: CQRS Pattern Implementation

use async_trait::async_trait;
use std::sync::Arc;
use uuid::Uuid;

use super::commands::{
    Command, CommandHandler, CommandResult,
    CreateResearchWorkflowCommand, StartWorkflowExecutionCommand,
    CreateTaskCommand, CompleteTaskCommand, CompleteWorkflowCommand, FailWorkflowCommand,
};
use super::queries::{
    Query, QueryHandler,
    GetResearchWorkflowQuery, GetWorkflowListQuery, GetWorkflowStatsQuery,
    GetTasksByWorkflowQuery, SearchWorkflowsQuery,
};
use super::read_models::{
    ReadModelStore, ResearchWorkflowReadModel, WorkflowListReadModel,
    WorkflowStatsReadModel, TaskReadModel,
};
use super::error::{CQRSError, CQRSResult};
use crate::event_store::{
    EventStore, ResearchWorkflowAggregate, AggregateRoot, DomainEvent,
    EventFactory, ResearchWorkflowEvent,
};

/// Shared aggregate persistence operations — eliminates load/save duplication
/// across all command handlers. Each handler composes this instead of
/// re-implementing the same event-store round-trip.
struct AggregatePersistence {
    event_store: Arc<EventStore>,
}

impl AggregatePersistence {
    fn new(event_store: Arc<EventStore>) -> Self {
        Self { event_store }
    }

    async fn load(&self, workflow_id: Uuid) -> CQRSResult<ResearchWorkflowAggregate> {
        let events = self
            .event_store
            .read_events(workflow_id, None, None)
            .await
            .map_err(|e| CQRSError::event_store_error(e.to_string()))?;

        if events.is_empty() {
            return Err(CQRSError::not_found(format!("Workflow {}", workflow_id)));
        }

        let mut aggregate = ResearchWorkflowAggregate::restore_from_state(
            workflow_id,
            crate::event_store::aggregates::ResearchWorkflowState::default(),
            0,
        );

        for event in events {
            if let Ok(workflow_event) = event.serialize() {
                if let Ok(deserialized_event) = serde_json::from_value::<ResearchWorkflowEvent>(workflow_event) {
                    aggregate.apply_event(&deserialized_event);
                }
            }
        }

        Ok(aggregate)
    }

    async fn save(&self, aggregate: &mut ResearchWorkflowAggregate) -> CQRSResult<u64> {
        let uncommitted = aggregate.get_uncommitted_events();
        if uncommitted.is_empty() {
            return Ok(aggregate.get_version());
        }

        let expected_version = aggregate.get_version() - uncommitted.len() as u64;
        let events: Vec<Box<dyn DomainEvent>> = uncommitted
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn DomainEvent>)
            .collect();

        let version = self
            .event_store
            .append_events(aggregate.get_id(), events, Some(expected_version))
            .await
            .map_err(|e| CQRSError::event_store_error(e.to_string()))?;

        aggregate.mark_events_as_committed();
        Ok(version)
    }

    async fn load_mutate_save<F>(
        &self,
        workflow_id: Uuid,
        command_id: Uuid,
        mutate: F,
    ) -> CQRSResult<CommandResult>
    where
        F: FnOnce(&mut ResearchWorkflowAggregate) -> Result<(), crate::event_store::error::EventStoreError>,
    {
        let mut aggregate = self.load(workflow_id).await?;
        mutate(&mut aggregate).map_err(|e| CQRSError::event_store_error(e.to_string()))?;
        let version = self.save(&mut aggregate).await?;
        Ok(CommandResult::success(command_id, workflow_id, version))
    }
}

/// Command Handlers

/// Create research workflow command handler
pub struct CreateResearchWorkflowHandler {
    persistence: AggregatePersistence,
}

impl CreateResearchWorkflowHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<CreateResearchWorkflowCommand> for CreateResearchWorkflowHandler {
    async fn handle(&self, command: CreateResearchWorkflowCommand) -> CQRSResult<CommandResult> {
        let mut aggregate = ResearchWorkflowAggregate::create_workflow(
            command.workflow_id,
            command.name,
            command.query,
            command.methodology,
        )
        .map_err(|e| CQRSError::event_store_error(e.to_string()))?;

        let version = self.persistence.save(&mut aggregate).await?;

        Ok(CommandResult::success(
            command.command_id,
            command.workflow_id,
            version,
        ))
    }
}

/// Start workflow execution command handler
pub struct StartWorkflowExecutionHandler {
    persistence: AggregatePersistence,
}

impl StartWorkflowExecutionHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<StartWorkflowExecutionCommand> for StartWorkflowExecutionHandler {
    async fn handle(&self, command: StartWorkflowExecutionCommand) -> CQRSResult<CommandResult> {
        self.persistence.load_mutate_save(
            command.workflow_id,
            command.command_id,
            |agg| agg.start_execution(),
        ).await
    }
}

/// Create task command handler
pub struct CreateTaskHandler {
    persistence: AggregatePersistence,
}

impl CreateTaskHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<CreateTaskCommand> for CreateTaskHandler {
    async fn handle(&self, command: CreateTaskCommand) -> CQRSResult<CommandResult> {
        self.persistence.load_mutate_save(
            command.workflow_id,
            command.command_id,
            |agg| agg.create_task(command.task_id, command.task_type, command.agent_type),
        ).await
    }
}

/// Complete task command handler
pub struct CompleteTaskHandler {
    persistence: AggregatePersistence,
}

impl CompleteTaskHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<CompleteTaskCommand> for CompleteTaskHandler {
    async fn handle(&self, command: CompleteTaskCommand) -> CQRSResult<CommandResult> {
        self.persistence.load_mutate_save(
            command.workflow_id,
            command.command_id,
            |agg| agg.complete_task(command.task_id, command.results),
        ).await
    }
}

/// Complete workflow command handler
pub struct CompleteWorkflowHandler {
    persistence: AggregatePersistence,
}

impl CompleteWorkflowHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<CompleteWorkflowCommand> for CompleteWorkflowHandler {
    async fn handle(&self, command: CompleteWorkflowCommand) -> CQRSResult<CommandResult> {
        self.persistence.load_mutate_save(
            command.workflow_id,
            command.command_id,
            |agg| agg.complete_execution(command.results),
        ).await
    }
}

/// Fail workflow command handler
pub struct FailWorkflowHandler {
    persistence: AggregatePersistence,
}

impl FailWorkflowHandler {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self { persistence: AggregatePersistence::new(event_store) }
    }
}

#[async_trait]
impl CommandHandler<FailWorkflowCommand> for FailWorkflowHandler {
    async fn handle(&self, command: FailWorkflowCommand) -> CQRSResult<CommandResult> {
        self.persistence.load_mutate_save(
            command.workflow_id,
            command.command_id,
            |agg| agg.fail_execution(command.error_message),
        ).await
    }
}

/// Query Handlers

/// Get research workflow query handler
pub struct GetResearchWorkflowHandler {
    read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>,
}

impl GetResearchWorkflowHandler {
    pub fn new(read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>) -> Self {
        Self { read_model_store }
    }
}

#[async_trait]
impl QueryHandler<GetResearchWorkflowQuery> for GetResearchWorkflowHandler {
    async fn handle(&self, query: GetResearchWorkflowQuery) -> CQRSResult<Option<ResearchWorkflowReadModel>> {
        let store = self.read_model_store.read().await;
        store.get_workflow(query.workflow_id).await
    }
}

/// Get workflow list query handler
pub struct GetWorkflowListHandler {
    read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>,
}

impl GetWorkflowListHandler {
    pub fn new(read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>) -> Self {
        Self { read_model_store }
    }
}

#[async_trait]
impl QueryHandler<GetWorkflowListQuery> for GetWorkflowListHandler {
    async fn handle(&self, query: GetWorkflowListQuery) -> CQRSResult<WorkflowListReadModel> {
        let store = self.read_model_store.read().await;
        store
            .get_workflow_list(
                query.page,
                query.page_size,
                query.status_filter,
                query.search_query,
                query.sort_by,
                query.sort_order,
            )
            .await
    }
}

/// Get workflow stats query handler
pub struct GetWorkflowStatsHandler {
    read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>,
}

impl GetWorkflowStatsHandler {
    pub fn new(read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>) -> Self {
        Self { read_model_store }
    }
}

#[async_trait]
impl QueryHandler<GetWorkflowStatsQuery> for GetWorkflowStatsHandler {
    async fn handle(&self, query: GetWorkflowStatsQuery) -> CQRSResult<WorkflowStatsReadModel> {
        let store = self.read_model_store.read().await;
        store
            .get_workflow_stats(
                query.date_range_start,
                query.date_range_end,
                query.group_by,
            )
            .await
    }
}

/// Get tasks by workflow query handler
pub struct GetTasksByWorkflowHandler {
    read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>,
}

impl GetTasksByWorkflowHandler {
    pub fn new(read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>) -> Self {
        Self { read_model_store }
    }
}

#[async_trait]
impl QueryHandler<GetTasksByWorkflowQuery> for GetTasksByWorkflowHandler {
    async fn handle(&self, query: GetTasksByWorkflowQuery) -> CQRSResult<Vec<TaskReadModel>> {
        let store = self.read_model_store.read().await;
        store
            .get_tasks_by_workflow(query.workflow_id, query.status_filter)
            .await
    }
}

/// Search workflows query handler
pub struct SearchWorkflowsHandler {
    read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>,
}

impl SearchWorkflowsHandler {
    pub fn new(read_model_store: Arc<tokio::sync::RwLock<dyn ReadModelStore>>) -> Self {
        Self { read_model_store }
    }
}

#[async_trait]
impl QueryHandler<SearchWorkflowsQuery> for SearchWorkflowsHandler {
    async fn handle(&self, query: SearchWorkflowsQuery) -> CQRSResult<WorkflowListReadModel> {
        let store = self.read_model_store.read().await;
        store
            .search_workflows(
                query.search_term,
                query.page,
                query.page_size,
                query.filters,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cqrs::read_models::MockReadModelStore;
    use crate::event_store::{EventStoreConfig, JsonEventSerializer};
    use crate::event_store::events::ResearchMethodology;
    use std::sync::Arc;

    async fn setup_test_event_store() -> Arc<EventStore> {
        // In real tests, you'd use a test database
        let database_url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgresql://test:test@localhost/test_fdr".to_string());
        
        let pool = sqlx::PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");
        
        let config = EventStoreConfig::default();
        let serializer = Arc::new(JsonEventSerializer::new());
        
        Arc::new(EventStore::new(pool, config, serializer))
    }

    #[tokio::test]
    async fn test_create_workflow_handler() {
        let event_store = setup_test_event_store().await;
        let handler = CreateResearchWorkflowHandler::new(event_store);
        
        let methodology = ResearchMethodology {
            name: "Test Method".to_string(),
            steps: vec!["Step 1".to_string()],
            ai_agents: vec!["agent1".to_string()],
            estimated_duration_minutes: 30,
        };
        
        let command = CreateResearchWorkflowCommand {
            command_id: Uuid::new_v4(),
            workflow_id: Uuid::new_v4(),
            name: "Test Workflow".to_string(),
            query: "Test Query".to_string(),
            methodology,
            correlation_id: None,
        };
        
        let result = handler.handle(command).await;
        assert!(result.is_ok());
        
        let command_result = result.unwrap();
        assert!(command_result.success);
        assert_eq!(command_result.version, Some(1));
    }

    #[tokio::test]
    async fn test_get_workflow_handler() {
        let read_model_store = Arc::new(tokio::sync::RwLock::new(MockReadModelStore::new()));
        let handler = GetResearchWorkflowHandler::new(read_model_store);
        
        let query = GetResearchWorkflowQuery {
            query_id: Uuid::new_v4(),
            workflow_id: Uuid::new_v4(),
            include_tasks: true,
            correlation_id: None,
        };
        
        let result = handler.handle(query).await;
        assert!(result.is_ok());
        
        // Should return None for non-existent workflow
        let workflow = result.unwrap();
        assert!(workflow.is_none());
    }

    #[tokio::test]
    async fn test_get_workflow_list_handler() {
        let read_model_store = Arc::new(tokio::sync::RwLock::new(MockReadModelStore::new()));
        let handler = GetWorkflowListHandler::new(read_model_store);
        
        let query = GetWorkflowListQuery {
            query_id: Uuid::new_v4(),
            page: 1,
            page_size: 10,
            status_filter: None,
            search_query: None,
            sort_by: None,
            sort_order: None,
            correlation_id: None,
        };
        
        let result = handler.handle(query).await;
        assert!(result.is_ok());
        
        let workflow_list = result.unwrap();
        assert_eq!(workflow_list.workflows.len(), 0);
        assert_eq!(workflow_list.total_count, 0);
    }
}

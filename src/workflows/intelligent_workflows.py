"""
Intelligent Workflows System
Implements decision trees, protocol automation, template systems,
quality assurance, and completion tracking for clinical workflows
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class WorkflowTask:
    """Individual workflow task"""
    task_id: str
    name: str
    description: str
    task_type: str  # 'manual', 'automated', 'decision', 'approval'
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    input_requirements: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    automated_function: Optional[Callable] = None
    decision_logic: Optional[Dict] = None
    approval_required: bool = False
    estimated_duration: int = 30  # minutes
    assigned_to: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error_message: Optional[str] = None

@dataclass
class DecisionNode:
    """Decision tree node"""
    node_id: str
    condition: str  # Python expression to evaluate
    description: str
    true_branch: Optional[str] = None  # Next node if condition is true
    false_branch: Optional[str] = None  # Next node if condition is false
    leaf_action: Optional[str] = None  # Action if this is a leaf node
    confidence_threshold: float = 0.8
    override_allowed: bool = False

@dataclass
class WorkflowTemplate:
    """Workflow template definition"""
    template_id: str
    name: str
    description: str
    category: str  # 'diagnostic', 'treatment', 'screening', 'follow-up'
    version: str
    tasks: List[WorkflowTask]
    decision_tree: Dict[str, DecisionNode] = field(default_factory=dict)
    entry_point: str = "start"
    quality_checks: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    approved: bool = False

@dataclass
class WorkflowInstance:
    """Active workflow instance"""
    instance_id: str
    template_id: str
    patient_id: str
    current_task: Optional[str] = None
    current_node: str = "start"
    context_data: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    assigned_clinician: Optional[str] = None
    quality_score: Optional[float] = None
    completion_percentage: float = 0.0
    alerts: List[str] = field(default_factory=list)

class DecisionEngine:
    """Clinical decision tree engine"""

    def __init__(self):
        self.decision_trees = {}
        self.evaluation_context = {}

    def add_decision_tree(self, tree_id: str, nodes: Dict[str, DecisionNode]):
        """Add a decision tree"""
        self.decision_trees[tree_id] = nodes
        logger.info(f"Added decision tree: {tree_id} with {len(nodes)} nodes")

    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition string"""
        try:
            # Create safe evaluation environment
            safe_globals = {
                '__builtins__': {},
                'abs': abs, 'max': max, 'min': min, 'len': len,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'round': round, 'sum': sum, 'any': any, 'all': all
            }

            # Add context variables
            safe_globals.update(context)

            # Evaluate condition
            result = eval(condition, safe_globals)
            return bool(result)

        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def traverse_decision_tree(self, tree_id: str, context: Dict[str, Any],
                             start_node: str = "start") -> Dict[str, Any]:
        """Traverse decision tree and return recommendation"""

        if tree_id not in self.decision_trees:
            raise ValueError(f"Decision tree {tree_id} not found")

        tree = self.decision_trees[tree_id]
        current_node_id = start_node
        path = []

        while current_node_id:
            if current_node_id not in tree:
                logger.error(f"Node {current_node_id} not found in tree {tree_id}")
                break

            node = tree[current_node_id]
            path.append(node.node_id)

            # If this is a leaf node, return the action
            if node.leaf_action:
                return {
                    'action': node.leaf_action,
                    'path': path,
                    'final_node': node.node_id,
                    'description': node.description,
                    'confidence': 1.0  # Leaf nodes have full confidence
                }

            # Evaluate condition
            condition_result = self.evaluate_condition(node.condition, context)

            # Move to next node
            if condition_result:
                current_node_id = node.true_branch
            else:
                current_node_id = node.false_branch

        return {
            'action': 'no_decision',
            'path': path,
            'error': 'Decision path incomplete'
        }

    def create_diabetic_foot_decision_tree(self) -> Dict[str, DecisionNode]:
        """Create decision tree for diabetic foot screening workflow"""

        nodes = {
            'start': DecisionNode(
                node_id='start',
                condition="diabetes_present == True",
                description="Check if patient has diabetes",
                true_branch='assess_risk_factors',
                false_branch='standard_foot_care'
            ),

            'assess_risk_factors': DecisionNode(
                node_id='assess_risk_factors',
                condition="neuropathy_present == True or previous_ulcer == True",
                description="Assess high-risk factors",
                true_branch='high_risk_protocol',
                false_branch='moderate_risk_check'
            ),

            'moderate_risk_check': DecisionNode(
                node_id='moderate_risk_check',
                condition="foot_deformities > 1 or vascular_disease == True",
                description="Check for moderate risk factors",
                true_branch='moderate_risk_protocol',
                false_branch='low_risk_protocol'
            ),

            'high_risk_protocol': DecisionNode(
                node_id='high_risk_protocol',
                condition="",
                description="High-risk diabetic foot protocol",
                leaf_action='initiate_high_risk_diabetic_foot_protocol'
            ),

            'moderate_risk_protocol': DecisionNode(
                node_id='moderate_risk_protocol',
                condition="",
                description="Moderate-risk diabetic foot protocol",
                leaf_action='initiate_moderate_risk_diabetic_foot_protocol'
            ),

            'low_risk_protocol': DecisionNode(
                node_id='low_risk_protocol',
                condition="",
                description="Low-risk diabetic foot protocol",
                leaf_action='initiate_low_risk_diabetic_foot_protocol'
            ),

            'standard_foot_care': DecisionNode(
                node_id='standard_foot_care',
                condition="",
                description="Standard foot care protocol",
                leaf_action='initiate_standard_foot_care_protocol'
            )
        }

        return nodes

    def create_treatment_decision_tree(self) -> Dict[str, DecisionNode]:
        """Create decision tree for treatment selection"""

        nodes = {
            'start': DecisionNode(
                node_id='start',
                condition="pain_level >= 7",
                description="Assess pain severity",
                true_branch='severe_pain_branch',
                false_branch='mild_moderate_pain_branch'
            ),

            'severe_pain_branch': DecisionNode(
                node_id='severe_pain_branch',
                condition="conservative_treatment_failed == True",
                description="Check if conservative treatment failed",
                true_branch='consider_injection_surgery',
                false_branch='intensive_conservative'
            ),

            'mild_moderate_pain_branch': DecisionNode(
                node_id='mild_moderate_pain_branch',
                condition="functional_limitation >= 3",
                description="Assess functional impact",
                true_branch='structured_conservative',
                false_branch='basic_conservative'
            ),

            'consider_injection_surgery': DecisionNode(
                node_id='consider_injection_surgery',
                condition="surgical_candidate == True",
                description="Evaluate surgical candidacy",
                true_branch='surgical_consultation',
                false_branch='injection_therapy'
            ),

            'surgical_consultation': DecisionNode(
                node_id='surgical_consultation',
                condition="",
                description="Surgical consultation recommended",
                leaf_action='schedule_surgical_consultation'
            ),

            'injection_therapy': DecisionNode(
                node_id='injection_therapy',
                condition="",
                description="Injection therapy recommended",
                leaf_action='schedule_injection_therapy'
            ),

            'intensive_conservative': DecisionNode(
                node_id='intensive_conservative',
                condition="",
                description="Intensive conservative treatment",
                leaf_action='initiate_intensive_conservative_treatment'
            ),

            'structured_conservative': DecisionNode(
                node_id='structured_conservative',
                condition="",
                description="Structured conservative treatment",
                leaf_action='initiate_structured_conservative_treatment'
            ),

            'basic_conservative': DecisionNode(
                node_id='basic_conservative',
                condition="",
                description="Basic conservative treatment",
                leaf_action='initiate_basic_conservative_treatment'
            )
        }

        return nodes

class WorkflowAutomation:
    """Automated workflow execution engine"""

    def __init__(self):
        self.automated_tasks = {}
        self.task_registry = {}

    def register_automated_task(self, task_type: str, function: Callable):
        """Register an automated task function"""
        self.automated_tasks[task_type] = function
        logger.info(f"Registered automated task: {task_type}")

    def execute_automated_task(self, task: WorkflowTask, context: Dict) -> Dict[str, Any]:
        """Execute an automated task"""

        if task.task_type not in self.automated_tasks:
            raise ValueError(f"No automation available for task type: {task.task_type}")

        try:
            task.status = WorkflowStatus.IN_PROGRESS
            task.started_at = datetime.now()

            # Execute the automated function
            function = self.automated_tasks[task.task_type]
            result = function(task, context)

            task.status = WorkflowStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            return result

        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Automated task {task.task_id} failed: {e}")
            raise

    def create_automated_functions(self):
        """Create standard automated functions"""

        def automated_risk_assessment(task: WorkflowTask, context: Dict) -> Dict:
            """Automated risk assessment task"""
            patient_data = context.get('patient_data', {})
            medical_conditions = context.get('medical_conditions', {})

            # Simple risk scoring
            risk_score = 0
            risk_factors = []

            if patient_data.get('diabetes'):
                risk_score += 2
                risk_factors.append('Diabetes')

            if patient_data.get('age', 0) > 65:
                risk_score += 1
                risk_factors.append('Advanced age')

            detected_conditions = len([c for c in medical_conditions.values() if getattr(c, 'detected', False)])
            risk_score += detected_conditions * 0.5

            risk_level = 'low'
            if risk_score >= 4:
                risk_level = 'high'
            elif risk_score >= 2:
                risk_level = 'moderate'

            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'completed_at': datetime.now().isoformat()
            }

        def automated_imaging_order(task: WorkflowTask, context: Dict) -> Dict:
            """Automated imaging order task"""
            conditions = context.get('medical_conditions', {})
            patient_data = context.get('patient_data', {})

            imaging_orders = []

            # Check conditions that require imaging
            for name, condition in conditions.items():
                if not getattr(condition, 'detected', False):
                    continue

                if 'fracture' in name.lower():
                    imaging_orders.append('X-ray: AP and lateral foot')
                elif 'arthritis' in name.lower() or 'rigidus' in name.lower():
                    imaging_orders.append('X-ray: Weight-bearing views')
                elif 'vascular' in name.lower():
                    imaging_orders.append('Doppler ultrasound')

            return {
                'imaging_orders': imaging_orders,
                'priority': 'routine' if len(imaging_orders) <= 2 else 'urgent',
                'ordering_physician': context.get('clinician_id', 'Dr. System'),
                'completed_at': datetime.now().isoformat()
            }

        def automated_documentation(task: WorkflowTask, context: Dict) -> Dict:
            """Automated documentation task"""
            template = task.input_requirements.get('template', 'standard')

            documentation = {
                'assessment_date': datetime.now().isoformat(),
                'patient_id': context.get('patient_id'),
                'clinician': context.get('clinician_id', 'Dr. System'),
                'findings': context.get('medical_conditions', {}),
                'recommendations': context.get('recommendations', []),
                'template_used': template
            }

            return documentation

        # Register the automated functions
        self.register_automated_task('risk_assessment', automated_risk_assessment)
        self.register_automated_task('imaging_order', automated_imaging_order)
        self.register_automated_task('documentation', automated_documentation)

class QualityAssurance:
    """Quality assurance and validation system"""

    def __init__(self):
        self.quality_rules = []
        self.validation_functions = {}

    def add_quality_rule(self, rule_id: str, description: str,
                        validation_function: Callable, severity: str = 'warning'):
        """Add a quality assurance rule"""
        rule = {
            'rule_id': rule_id,
            'description': description,
            'validation_function': validation_function,
            'severity': severity  # 'info', 'warning', 'error', 'critical'
        }
        self.quality_rules.append(rule)

    def validate_workflow_instance(self, instance: WorkflowInstance,
                                 template: WorkflowTemplate) -> Dict[str, Any]:
        """Validate a workflow instance against quality rules"""

        validation_results = {
            'overall_score': 0.0,
            'passed_checks': 0,
            'total_checks': len(self.quality_rules),
            'issues': [],
            'warnings': [],
            'errors': []
        }

        for rule in self.quality_rules:
            try:
                result = rule['validation_function'](instance, template)

                if result['passed']:
                    validation_results['passed_checks'] += 1
                else:
                    issue = {
                        'rule_id': rule['rule_id'],
                        'description': rule['description'],
                        'severity': rule['severity'],
                        'details': result.get('details', ''),
                        'recommendation': result.get('recommendation', '')
                    }

                    validation_results['issues'].append(issue)

                    if rule['severity'] == 'warning':
                        validation_results['warnings'].append(issue)
                    elif rule['severity'] in ['error', 'critical']:
                        validation_results['errors'].append(issue)

            except Exception as e:
                logger.error(f"Quality rule {rule['rule_id']} failed: {e}")

        # Calculate overall quality score
        if validation_results['total_checks'] > 0:
            validation_results['overall_score'] = (
                validation_results['passed_checks'] / validation_results['total_checks']
            )

        return validation_results

    def create_standard_quality_rules(self):
        """Create standard quality assurance rules"""

        def check_completion_percentage(instance: WorkflowInstance, template: WorkflowTemplate) -> Dict:
            """Check if workflow completion percentage is reasonable"""
            expected_completion = 100.0 if instance.status == WorkflowStatus.COMPLETED else 50.0

            if instance.completion_percentage < expected_completion * 0.8:
                return {
                    'passed': False,
                    'details': f'Completion percentage ({instance.completion_percentage:.1f}%) lower than expected',
                    'recommendation': 'Review workflow progress and update completion tracking'
                }
            return {'passed': True}

        def check_critical_tasks_completed(instance: WorkflowInstance, template: WorkflowTemplate) -> Dict:
            """Check if critical priority tasks are completed"""
            critical_tasks = [t for t in template.tasks if t.priority == TaskPriority.CRITICAL]

            for task in critical_tasks:
                if task.task_id not in instance.task_results:
                    return {
                        'passed': False,
                        'details': f'Critical task {task.name} not completed',
                        'recommendation': 'Complete all critical priority tasks before finalizing'
                    }
            return {'passed': True}

        def check_documentation_completeness(instance: WorkflowInstance, template: WorkflowTemplate) -> Dict:
            """Check if required documentation is complete"""
            doc_tasks = [t for t in template.tasks if 'documentation' in t.task_type.lower()]

            incomplete_docs = []
            for task in doc_tasks:
                if task.task_id in instance.task_results:
                    result = instance.task_results[task.task_id]
                    if not result or len(str(result).strip()) < 10:
                        incomplete_docs.append(task.name)
                else:
                    incomplete_docs.append(task.name)

            if incomplete_docs:
                return {
                    'passed': False,
                    'details': f'Incomplete documentation: {", ".join(incomplete_docs)}',
                    'recommendation': 'Complete all required documentation fields'
                }
            return {'passed': True}

        def check_timeline_adherence(instance: WorkflowInstance, template: WorkflowTemplate) -> Dict:
            """Check if workflow is completed within reasonable timeline"""
            if instance.completed_at and instance.started_at:
                duration = (instance.completed_at - instance.started_at).total_seconds() / 3600  # hours
                estimated_duration = sum(t.estimated_duration for t in template.tasks) / 60  # convert to hours

                if duration > estimated_duration * 2:
                    return {
                        'passed': False,
                        'details': f'Workflow took {duration:.1f}h, expected ~{estimated_duration:.1f}h',
                        'recommendation': 'Review workflow efficiency and identify bottlenecks'
                    }
            return {'passed': True}

        # Register quality rules
        self.add_quality_rule('completion_percentage', 'Completion percentage validation',
                            check_completion_percentage, 'warning')
        self.add_quality_rule('critical_tasks', 'Critical tasks completion check',
                            check_critical_tasks_completed, 'error')
        self.add_quality_rule('documentation', 'Documentation completeness check',
                            check_documentation_completeness, 'warning')
        self.add_quality_rule('timeline', 'Timeline adherence check',
                            check_timeline_adherence, 'info')

class WorkflowEngine:
    """Main workflow execution engine"""

    def __init__(self):
        self.templates = {}
        self.instances = {}
        self.decision_engine = DecisionEngine()
        self.automation = WorkflowAutomation()
        self.quality_assurance = QualityAssurance()

        # Initialize components
        self._initialize_decision_trees()
        self.automation.create_automated_functions()
        self.quality_assurance.create_standard_quality_rules()

        logger.info("Workflow Engine initialized")

    def _initialize_decision_trees(self):
        """Initialize standard decision trees"""

        # Add diabetic foot decision tree
        diabetic_tree = self.decision_engine.create_diabetic_foot_decision_tree()
        self.decision_engine.add_decision_tree('diabetic_foot_screening', diabetic_tree)

        # Add treatment decision tree
        treatment_tree = self.decision_engine.create_treatment_decision_tree()
        self.decision_engine.add_decision_tree('treatment_selection', treatment_tree)

    def create_workflow_template(self, template_data: Dict) -> WorkflowTemplate:
        """Create a new workflow template"""

        template = WorkflowTemplate(
            template_id=template_data['template_id'],
            name=template_data['name'],
            description=template_data['description'],
            category=template_data.get('category', 'general'),
            version=template_data.get('version', '1.0'),
            tasks=[],
            created_by=template_data.get('created_by', 'system')
        )

        # Create tasks
        for task_data in template_data.get('tasks', []):
            task = WorkflowTask(
                task_id=task_data['task_id'],
                name=task_data['name'],
                description=task_data['description'],
                task_type=task_data.get('task_type', 'manual'),
                priority=TaskPriority(task_data.get('priority', 2)),
                dependencies=task_data.get('dependencies', []),
                estimated_duration=task_data.get('estimated_duration', 30)
            )
            template.tasks.append(task)

        self.templates[template.template_id] = template
        logger.info(f"Created workflow template: {template.name}")

        return template

    def start_workflow_instance(self, template_id: str, patient_id: str,
                              context_data: Dict, clinician_id: Optional[str] = None) -> WorkflowInstance:
        """Start a new workflow instance"""

        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]

        instance = WorkflowInstance(
            instance_id=f"{template_id}_{patient_id}_{int(datetime.now().timestamp())}",
            template_id=template_id,
            patient_id=patient_id,
            context_data=context_data,
            assigned_clinician=clinician_id,
            status=WorkflowStatus.IN_PROGRESS
        )

        self.instances[instance.instance_id] = instance

        # Start first task
        self._advance_workflow(instance)

        logger.info(f"Started workflow instance: {instance.instance_id}")
        return instance

    def _advance_workflow(self, instance: WorkflowInstance):
        """Advance workflow to next available task"""

        template = self.templates[instance.template_id]

        # Find next available task
        available_tasks = self._get_available_tasks(instance, template)

        if not available_tasks:
            # No more tasks, check completion
            if self._is_workflow_complete(instance, template):
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                instance.completion_percentage = 100.0

                # Run quality assurance
                qa_results = self.quality_assurance.validate_workflow_instance(instance, template)
                instance.quality_score = qa_results['overall_score']

                logger.info(f"Workflow {instance.instance_id} completed with quality score {instance.quality_score:.2f}")
            return

        # Select next task (highest priority first)
        next_task = max(available_tasks, key=lambda t: t.priority.value)
        instance.current_task = next_task.task_id

        # If task is automated, execute it
        if next_task.task_type in self.automation.automated_tasks:
            try:
                result = self.automation.execute_automated_task(next_task, instance.context_data)
                self.complete_task(instance.instance_id, next_task.task_id, result)
            except Exception as e:
                logger.error(f"Automated task failed: {e}")
                instance.alerts.append(f"Automated task {next_task.name} failed: {str(e)}")

    def _get_available_tasks(self, instance: WorkflowInstance, template: WorkflowTemplate) -> List[WorkflowTask]:
        """Get tasks that are available to execute"""

        available_tasks = []

        for task in template.tasks:
            # Skip completed tasks
            if task.task_id in instance.task_results:
                continue

            # Check dependencies
            dependencies_met = all(
                dep_id in instance.task_results for dep_id in task.dependencies
            )

            if dependencies_met:
                available_tasks.append(task)

        return available_tasks

    def _is_workflow_complete(self, instance: WorkflowInstance, template: WorkflowTemplate) -> bool:
        """Check if workflow is complete"""

        # All required tasks must be completed
        required_tasks = [t for t in template.tasks if t.priority != TaskPriority.LOW]

        for task in required_tasks:
            if task.task_id not in instance.task_results:
                return False

        return True

    def complete_task(self, instance_id: str, task_id: str, result: Dict[str, Any],
                     clinician_id: Optional[str] = None):
        """Complete a workflow task"""

        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance {instance_id} not found")

        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]

        # Find the task
        task = next((t for t in template.tasks if t.task_id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found in template")

        # Record result
        instance.task_results[task_id] = result

        # Update completion percentage
        completed_tasks = len(instance.task_results)
        total_tasks = len(template.tasks)
        instance.completion_percentage = (completed_tasks / total_tasks) * 100

        # Update context with task results
        instance.context_data[f'task_{task_id}_result'] = result

        # Advance workflow
        self._advance_workflow(instance)

        logger.info(f"Completed task {task_id} in workflow {instance_id}")

    def execute_decision_logic(self, instance_id: str, tree_id: str) -> Dict[str, Any]:
        """Execute decision tree logic"""

        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance {instance_id} not found")

        instance = self.instances[instance_id]

        # Execute decision tree
        decision_result = self.decision_engine.traverse_decision_tree(
            tree_id, instance.context_data
        )

        # Store decision result
        instance.context_data[f'decision_{tree_id}'] = decision_result

        return decision_result

    def get_workflow_status(self, instance_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status"""

        if instance_id not in self.instances:
            raise ValueError(f"Workflow instance {instance_id} not found")

        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]

        # Get current task info
        current_task_info = None
        if instance.current_task:
            current_task = next((t for t in template.tasks if t.task_id == instance.current_task), None)
            if current_task:
                current_task_info = {
                    'task_id': current_task.task_id,
                    'name': current_task.name,
                    'description': current_task.description,
                    'priority': current_task.priority.name,
                    'estimated_duration': current_task.estimated_duration
                }

        # Get pending tasks
        pending_tasks = self._get_available_tasks(instance, template)
        pending_task_info = [
            {
                'task_id': t.task_id,
                'name': t.name,
                'priority': t.priority.name,
                'dependencies': t.dependencies
            }
            for t in pending_tasks
        ]

        return {
            'instance_id': instance.instance_id,
            'template_name': template.name,
            'patient_id': instance.patient_id,
            'status': instance.status.value,
            'completion_percentage': instance.completion_percentage,
            'current_task': current_task_info,
            'pending_tasks': pending_task_info,
            'completed_tasks': len(instance.task_results),
            'total_tasks': len(template.tasks),
            'quality_score': instance.quality_score,
            'alerts': instance.alerts,
            'started_at': instance.started_at.isoformat(),
            'completed_at': instance.completed_at.isoformat() if instance.completed_at else None
        }

    def create_diabetic_foot_screening_template(self) -> WorkflowTemplate:
        """Create template for diabetic foot screening workflow"""

        template_data = {
            'template_id': 'diabetic_foot_screening',
            'name': 'Diabetic Foot Screening Protocol',
            'description': 'Comprehensive diabetic foot screening and risk assessment workflow',
            'category': 'screening',
            'version': '2.0',
            'tasks': [
                {
                    'task_id': 'patient_history',
                    'name': 'Patient History Collection',
                    'description': 'Collect relevant medical history and risk factors',
                    'task_type': 'manual',
                    'priority': 4,
                    'estimated_duration': 15
                },
                {
                    'task_id': 'foot_examination',
                    'name': 'Physical Foot Examination',
                    'description': 'Comprehensive foot and ankle examination',
                    'task_type': 'manual',
                    'priority': 5,
                    'dependencies': ['patient_history'],
                    'estimated_duration': 20
                },
                {
                    'task_id': 'risk_assessment',
                    'name': 'Automated Risk Assessment',
                    'description': 'Calculate diabetic foot risk score',
                    'task_type': 'risk_assessment',
                    'priority': 4,
                    'dependencies': ['patient_history', 'foot_examination'],
                    'estimated_duration': 5
                },
                {
                    'task_id': 'decision_tree_evaluation',
                    'name': 'Protocol Decision',
                    'description': 'Determine appropriate care protocol',
                    'task_type': 'decision',
                    'priority': 5,
                    'dependencies': ['risk_assessment'],
                    'estimated_duration': 2
                },
                {
                    'task_id': 'imaging_order',
                    'name': 'Imaging Orders',
                    'description': 'Order appropriate imaging studies if needed',
                    'task_type': 'imaging_order',
                    'priority': 3,
                    'dependencies': ['decision_tree_evaluation'],
                    'estimated_duration': 5
                },
                {
                    'task_id': 'patient_education',
                    'name': 'Patient Education',
                    'description': 'Provide diabetic foot care education',
                    'task_type': 'manual',
                    'priority': 3,
                    'dependencies': ['decision_tree_evaluation'],
                    'estimated_duration': 20
                },
                {
                    'task_id': 'documentation',
                    'name': 'Clinical Documentation',
                    'description': 'Complete clinical documentation',
                    'task_type': 'documentation',
                    'priority': 4,
                    'dependencies': ['patient_education'],
                    'estimated_duration': 10
                },
                {
                    'task_id': 'follow_up_scheduling',
                    'name': 'Follow-up Scheduling',
                    'description': 'Schedule appropriate follow-up appointments',
                    'task_type': 'manual',
                    'priority': 3,
                    'dependencies': ['documentation'],
                    'estimated_duration': 5
                }
            ]
        }

        return self.create_workflow_template(template_data)

    def create_treatment_planning_template(self) -> WorkflowTemplate:
        """Create template for treatment planning workflow"""

        template_data = {
            'template_id': 'treatment_planning',
            'name': 'Treatment Planning Protocol',
            'description': 'Comprehensive treatment planning and selection workflow',
            'category': 'treatment',
            'version': '2.0',
            'tasks': [
                {
                    'task_id': 'condition_assessment',
                    'name': 'Condition Assessment',
                    'description': 'Assess diagnosed conditions and severity',
                    'task_type': 'manual',
                    'priority': 5,
                    'estimated_duration': 15
                },
                {
                    'task_id': 'treatment_decision',
                    'name': 'Treatment Decision Tree',
                    'description': 'Determine optimal treatment approach',
                    'task_type': 'decision',
                    'priority': 5,
                    'dependencies': ['condition_assessment'],
                    'estimated_duration': 5
                },
                {
                    'task_id': 'treatment_planning',
                    'name': 'Treatment Plan Development',
                    'description': 'Develop comprehensive treatment plan',
                    'task_type': 'manual',
                    'priority': 4,
                    'dependencies': ['treatment_decision'],
                    'estimated_duration': 25
                },
                {
                    'task_id': 'patient_consent',
                    'name': 'Patient Consent',
                    'description': 'Obtain informed consent for treatment',
                    'task_type': 'manual',
                    'priority': 5,
                    'dependencies': ['treatment_planning'],
                    'estimated_duration': 15
                },
                {
                    'task_id': 'treatment_documentation',
                    'name': 'Treatment Documentation',
                    'description': 'Document treatment plan and rationale',
                    'task_type': 'documentation',
                    'priority': 4,
                    'dependencies': ['patient_consent'],
                    'estimated_duration': 10
                }
            ]
        }

        return self.create_workflow_template(template_data)

    def export_workflow_analytics(self, time_period: int = 30) -> Dict[str, Any]:
        """Export workflow analytics for the specified time period (days)"""

        cutoff_date = datetime.now() - timedelta(days=time_period)
        recent_instances = [
            instance for instance in self.instances.values()
            if instance.started_at >= cutoff_date
        ]

        analytics = {
            'period_days': time_period,
            'total_workflows': len(recent_instances),
            'completed_workflows': len([i for i in recent_instances if i.status == WorkflowStatus.COMPLETED]),
            'failed_workflows': len([i for i in recent_instances if i.status == WorkflowStatus.FAILED]),
            'average_completion_time': 0,
            'average_quality_score': 0,
            'template_usage': {},
            'common_bottlenecks': [],
            'quality_issues': []
        }

        # Calculate averages
        completed_workflows = [i for i in recent_instances if i.status == WorkflowStatus.COMPLETED and i.completed_at]

        if completed_workflows:
            completion_times = [
                (i.completed_at - i.started_at).total_seconds() / 3600  # hours
                for i in completed_workflows
            ]
            analytics['average_completion_time'] = sum(completion_times) / len(completion_times)

            quality_scores = [i.quality_score for i in completed_workflows if i.quality_score is not None]
            if quality_scores:
                analytics['average_quality_score'] = sum(quality_scores) / len(quality_scores)

        # Template usage statistics
        for instance in recent_instances:
            template_id = instance.template_id
            if template_id not in analytics['template_usage']:
                analytics['template_usage'][template_id] = 0
            analytics['template_usage'][template_id] += 1

        return analytics
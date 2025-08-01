"""
Virtual Healthcare System Simulation
A discrete event simulation of a hospital system with multiple departments
and complex patient flows.
"""

import random
import heapq
from enum import Enum, auto
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Set

# Global simulation time
current_time = 0.0

# Event queue
event_queue = []


class EventType(Enum):
    """Types of events in the simulation"""
    PATIENT_ARRIVAL = auto()
    SERVICE_COMPLETION = auto()
    RESOURCE_AVAILABLE = auto()
    SHIFT_CHANGE = auto()
    PATIENT_DETERIORATION = auto()


class PatientSeverity(Enum):
    """Severity levels for patients"""
    CRITICAL = 1
    SEVERE = 2
    MODERATE = 3
    MILD = 4
    ROUTINE = 5


class PatientStatus(Enum):
    """Status of patients in the system"""
    WAITING = auto()
    IN_SERVICE = auto()
    WAITING_FOR_RESOURCE = auto()
    COMPLETED = auto()
    LEFT_WITHOUT_TREATMENT = auto()


@dataclass
class Patient:
    """Patient entity with attributes and medical history"""
    id: int
    arrival_time: float
    severity: PatientSeverity
    requires_surgery: bool = False
    requires_imaging: bool = False
    requires_lab_tests: bool = False
    specialist_required: Optional[str] = None
    status: PatientStatus = PatientStatus.WAITING
    current_department: Optional[str] = None
    processing_history: List[Tuple[str, float, float]] = field(default_factory=list)

    def __lt__(self, other):
        """Comparison for priority queues - based on severity"""
        return self.severity.value < other.severity.value

    def add_history(self, department: str, start_time: float, end_time: float):
        """Record processing in a department"""
        self.processing_history.append((department, start_time, end_time))

    def total_time(self) -> float:
        """Calculate total time in system"""
        if self.status == PatientStatus.COMPLETED:
            return self.processing_history[-1][2] - self.arrival_time
        return current_time - self.arrival_time

    def waiting_time(self) -> float:
        """Calculate total waiting time"""
        service_time = sum(end - start for _, start, end in self.processing_history)
        return self.total_time() - service_time


@dataclass
class Event:
    """Event in the discrete event simulation"""
    time: float
    event_type: EventType
    department: str
    entity: Any = None
    resource: Optional[str] = None
    handler: Optional[Callable] = None

    def __lt__(self, other):
        """Comparison for priority queue"""
        return self.time < other.time


@dataclass
class Resource:
    """Represents staff or equipment resources"""
    name: str
    department: str
    available: bool = True
    skill_level: int = 1
    current_patient: Optional[Patient] = None
    shift_end_time: Optional[float] = None
    utilization_time: float = 0.0

    def assign(self, patient: Patient, current_time: float):
        """Assign resource to a patient"""
        self.available = False
        self.current_patient = patient
        self.utilization_start = current_time

    def release(self, current_time: float):
        """Release resource"""
        if not self.available:
            self.utilization_time += current_time - self.utilization_start
            self.available = True
            self.current_patient = None


class Department:
    """Base class for hospital departments"""

    def __init__(self, name: str):
        self.name = name
        self.waiting_queue = []  # Priority queue for patients
        self.resources = {}  # Resources assigned to this department
        self.patient_count = 0
        self.completed_patients = 0
        self.sub_departments = {}  # For hierarchical departments

    def add_resource(self, resource: Resource):
        """Add a resource to the department"""
        self.resources[resource.name] = resource

    def add_patient_to_queue(self, patient: Patient):
        """Add patient to waiting queue with priority based on severity"""
        patient.current_department = self.name
        heapq.heappush(self.waiting_queue, patient)
        self.patient_count += 1

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process a patient and return service time
        To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_patient")

    def available_resources(self) -> List[Resource]:
        """Return list of available resources"""
        return [r for r in self.resources.values() if r.available]

    def try_serve_next_patient(self, time: float) -> Optional[Event]:
        """Try to serve next patient if resources available"""
        if not self.waiting_queue:
            return None

        available = self.available_resources()
        if not available:
            return None

        # Get highest priority patient
        patient = heapq.heappop(self.waiting_queue)
        resource = available[0]  # Simple selection - can be improved

        # Assign resource to patient
        patient.status = PatientStatus.IN_SERVICE
        resource.assign(patient, time)

        # Calculate service time and schedule completion
        service_time = self.process_patient(patient, time)

        # Create service completion event
        return Event(
            time=time + service_time,
            event_type=EventType.SERVICE_COMPLETION,
            department=self.name,
            entity=patient,
            resource=resource.name
        )

    def complete_service(self, patient: Patient, resource_name: str, time: float) -> Patient:
        """Complete service for a patient and release resource"""
        resource = self.resources[resource_name]
        resource.release(time)

        patient.add_history(self.name,
                            patient.processing_history[-1][1] if patient.processing_history else patient.arrival_time,
                            time)
        self.completed_patients += 1
        return patient

    def add_sub_department(self, department):
        """Add a sub-department"""
        self.sub_departments[department.name] = department
        return department

    def statistics(self) -> Dict:
        """Return department statistics"""
        stats = {
            "name": self.name,
            "patients_processed": self.completed_patients,
            "current_queue_length": len(self.waiting_queue),
            "resource_utilization": {r.name: r.utilization_time / max(current_time, 1) for r in self.resources.values()}
        }

        # Add sub-department stats
        if self.sub_departments:
            stats["sub_departments"] = {name: dept.statistics() for name, dept in self.sub_departments.items()}

        return stats


class EmergencyDepartment(Department):
    """Emergency Department with triage capabilities"""

    def __init__(self):
        super().__init__("Emergency Department")
        self.triage_nurse_available = True
        self.triage_times = {
            PatientSeverity.CRITICAL: 5,
            PatientSeverity.SEVERE: 8,
            PatientSeverity.MODERATE: 12,
            PatientSeverity.MILD: 15,
            PatientSeverity.ROUTINE: 20
        }

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient in ED based on severity"""
        # Base service time varies by severity 
        base_time = {
            PatientSeverity.CRITICAL: random.uniform(45, 120),
            PatientSeverity.SEVERE: random.uniform(30, 90),
            PatientSeverity.MODERATE: random.uniform(20, 60),
            PatientSeverity.MILD: random.uniform(15, 45),
            PatientSeverity.ROUTINE: random.uniform(10, 30)
        }[patient.severity]

        # Add some variability
        service_time = base_time * (0.8 + 0.4 * random.random())

        # Add triage time
        service_time += self.triage_times[patient.severity]

        return service_time


class DiagnosticCenter(Department):
    """Diagnostic Center with imaging and laboratory sub-departments"""

    def __init__(self):
        super().__init__("Diagnostic Center")
        # Create sub-departments
        self.add_sub_department(ImagingDepartment())
        self.add_sub_department(LaboratoryDepartment())

    def process_patient(self, patient: Patient, time: float) -> float:
        """Route patient to appropriate sub-department"""
        # This is a coordination department - minimal processing time
        return 5.0  # Administrative processing


class ImagingDepartment(Department):
    """Imaging department with different imaging modalities"""

    def __init__(self):
        super().__init__("Imaging")
        self.modality_times = {
            "X-ray": (10, 20),
            "CT": (20, 40),
            "MRI": (30, 60),
            "Ultrasound": (15, 30)
        }

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient for imaging"""
        # Select a random imaging modality
        modality = random.choice(list(self.modality_times.keys()))
        min_time, max_time = self.modality_times[modality]

        # Calculate service time
        service_time = random.uniform(min_time, max_time)

        # Add preparation time
        prep_time = 5 + 5 * random.random()

        return service_time + prep_time


class LaboratoryDepartment(Department):
    """Laboratory for running tests"""

    def __init__(self):
        super().__init__("Laboratory")
        self.test_times = {
            "Blood": (30, 120),
            "Urine": (20, 60),
            "Tissue": (120, 240),
            "CSF": (60, 180)
        }

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient for lab tests"""
        # Select a random test type
        test_type = random.choice(list(self.test_times.keys()))
        min_time, max_time = self.test_times[test_type]

        # Calculate service time
        service_time = random.uniform(min_time, max_time)

        # Sample collection time
        collection_time = 5 + 10 * random.random()

        return service_time + collection_time


class SpecialistConsultation(Department):
    """Specialist consultation department"""

    def __init__(self):
        super().__init__("Specialist Consultation")
        self.specialties = ["Cardiology", "Neurology", "Orthopedics",
                            "Pediatrics", "Oncology", "General Surgery"]
        self.consultation_times = {
            "Cardiology": (15, 45),
            "Neurology": (20, 60),
            "Orthopedics": (15, 40),
            "Pediatrics": (10, 30),
            "Oncology": (20, 50),
            "General Surgery": (15, 35)
        }

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient for specialist consultation"""
        # If patient has a specific specialist required, use that
        if patient.specialist_required and patient.specialist_required in self.specialties:
            specialty = patient.specialist_required
        else:
            # Otherwise assign random specialty
            specialty = random.choice(self.specialties)

        min_time, max_time = self.consultation_times[specialty]

        # Calculate service time
        service_time = random.uniform(min_time, max_time)

        # Add review time for patient history
        review_time = 5 + 10 * random.random()

        return service_time + review_time


class SurgeryDepartment(Department):
    """Surgery department with operating rooms"""

    def __init__(self):
        super().__init__("Surgery")
        self.surgery_types = {
            "Minor": (30, 90),
            "Moderate": (60, 180),
            "Major": (120, 360),
            "Complex": (240, 480)
        }

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient for surgery"""
        # Determine surgery complexity based on patient severity
        surgery_complexity = {
            PatientSeverity.CRITICAL: "Complex",
            PatientSeverity.SEVERE: "Major",
            PatientSeverity.MODERATE: "Moderate",
            PatientSeverity.MILD: "Minor",
            PatientSeverity.ROUTINE: "Minor"
        }[patient.severity]

        min_time, max_time = self.surgery_types[surgery_complexity]

        # Calculate service time
        service_time = random.uniform(min_time, max_time)

        # Add preparation and anesthesia time
        prep_time = 30 + 30 * random.random()

        return service_time + prep_time


class RecoveryWard(Department):
    """Recovery ward for post-procedure monitoring"""

    def __init__(self):
        super().__init__("Recovery Ward")

    def process_patient(self, patient: Patient, time: float) -> float:
        """Process patient in recovery ward"""
        # Recovery time based on patient severity and procedures
        base_time = {
            PatientSeverity.CRITICAL: random.uniform(240, 1440),  # 4-24 hours
            PatientSeverity.SEVERE: random.uniform(180, 720),  # 3-12 hours
            PatientSeverity.MODERATE: random.uniform(120, 360),  # 2-6 hours
            PatientSeverity.MILD: random.uniform(60, 240),  # 1-4 hours
            PatientSeverity.ROUTINE: random.uniform(30, 120)  # 0.5-2 hours
        }[patient.severity]

        # Adjust for procedures performed
        if patient.requires_surgery:
            base_time *= 1.5

        # Add some variability based on patient recovery rate
        recovery_factor = 0.7 + 0.6 * random.random()  # 0.7 to 1.3

        return base_time * recovery_factor


class Hospital:
    """Main hospital system that coordinates departments"""

    def __init__(self):
        self.departments = {}
        self.patient_id_counter = 0
        self.completed_patients = []
        self.patient_pathways = []
        self.resource_allocation = {}

        # Create departments
        self.create_departments()

        # Create resources
        self.create_resources()

        # Statistics tracking
        self.statistics = {
            "patients_generated": 0,
            "patients_completed": 0,
            "patients_left": 0,
            "average_wait_time": 0,
            "average_total_time": 0,
            "department_stats": {}
        }

    def create_departments(self):
        """Create hospital departments"""
        # Main departments
        self.departments["ED"] = EmergencyDepartment()
        self.departments["Diagnostics"] = DiagnosticCenter()
        self.departments["Specialists"] = SpecialistConsultation()
        self.departments["Surgery"] = SurgeryDepartment()
        self.departments["Recovery"] = RecoveryWard()

        # Reference to sub-departments for easier access
        self.departments["Imaging"] = self.departments["Diagnostics"].sub_departments["Imaging"]
        self.departments["Laboratory"] = self.departments["Diagnostics"].sub_departments["Laboratory"]

    def create_resources(self):
        """Create and allocate resources to departments"""
        # Emergency Department Resources
        for i in range(3):
            self.add_resource(Resource(f"ED_Doctor_{i + 1}", "ED", skill_level=3))
        for i in range(5):
            self.add_resource(Resource(f"ED_Nurse_{i + 1}", "ED", skill_level=2))
        self.add_resource(Resource("ED_Triage_Nurse", "ED", skill_level=2))

        # Imaging Department Resources
        self.add_resource(Resource("X-ray_Machine", "Imaging", skill_level=1))
        self.add_resource(Resource("CT_Scanner", "Imaging", skill_level=2))
        self.add_resource(Resource("MRI_Machine", "Imaging", skill_level=3))
        self.add_resource(Resource("Ultrasound", "Imaging", skill_level=1))
        self.add_resource(Resource("Radiologist", "Imaging", skill_level=3))
        self.add_resource(Resource("Radiology_Tech_1", "Imaging", skill_level=2))
        self.add_resource(Resource("Radiology_Tech_2", "Imaging", skill_level=2))

        # Laboratory Resources
        for i in range(3):
            self.add_resource(Resource(f"Lab_Technician_{i + 1}", "Laboratory", skill_level=2))
        self.add_resource(Resource("Blood_Analyzer", "Laboratory", skill_level=1))
        self.add_resource(Resource("Centrifuge", "Laboratory", skill_level=1))

        # Specialist Resources
        specialties = ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology", "General Surgery"]
        for specialty in specialties:
            self.add_resource(Resource(f"{specialty}_Specialist", "Specialists", skill_level=4))

        # Surgery Resources
        for i in range(3):
            self.add_resource(Resource(f"Operating_Room_{i + 1}", "Surgery", skill_level=3))
        for i in range(4):
            self.add_resource(Resource(f"Surgeon_{i + 1}", "Surgery", skill_level=4))
        for i in range(6):
            self.add_resource(Resource(f"Surgery_Nurse_{i + 1}", "Surgery", skill_level=2))
        for i in range(3):
            self.add_resource(Resource(f"Anesthesiologist_{i + 1}", "Surgery", skill_level=4))

        # Recovery Resources
        for i in range(10):
            self.add_resource(Resource(f"Recovery_Bed_{i + 1}", "Recovery", skill_level=1))
        for i in range(4):
            self.add_resource(Resource(f"Recovery_Nurse_{i + 1}", "Recovery", skill_level=2))

    def add_resource(self, resource: Resource):
        """Add a resource to the appropriate department"""
        department = resource.department

        # Special case for diagnostic center sub-departments
        if department in ["Imaging", "Laboratory"]:
            self.departments[department].add_resource(resource)
        else:
            self.departments[department].add_resource(resource)

        # Track all resources
        self.resource_allocation[resource.name] = resource

    def generate_patient(self, time: float) -> Patient:
        """Generate a new patient with random attributes"""
        # Increment patient counter
        self.patient_id_counter += 1

        # Random severity with weighted distribution
        severity_weights = [0.05, 0.15, 0.3, 0.3, 0.2]  # Critical to Routine
        severity = random.choices(list(PatientSeverity), weights=severity_weights)[0]

        # Determine patient needs based on severity
        requires_surgery = random.random() < (0.8 if severity == PatientSeverity.CRITICAL else
                                              0.6 if severity == PatientSeverity.SEVERE else
                                              0.3 if severity == PatientSeverity.MODERATE else
                                              0.1 if severity == PatientSeverity.MILD else 0.05)

        requires_imaging = random.random() < 0.7
        requires_lab_tests = random.random() < 0.6

        # Determine if specialist is needed
        needs_specialist = random.random() < 0.5
        specialist_type = None
        if needs_specialist:
            specialist_type = random.choice(["Cardiology", "Neurology", "Orthopedics",
                                             "Pediatrics", "Oncology", "General Surgery"])

        # Create patient
        patient = Patient(
            id=self.patient_id_counter,
            arrival_time=time,
            severity=severity,
            requires_surgery=requires_surgery,
            requires_imaging=requires_imaging,
            requires_lab_tests=requires_lab_tests,
            specialist_required=specialist_type
        )

        self.statistics["patients_generated"] += 1

        return patient

    def determine_next_department(self, patient: Patient, current_dept: str) -> Optional[str]:
        """Determine the next department for a patient based on their needs and history"""
        # Create patient pathway if it hasn't been determined yet
        if not hasattr(patient, 'pathway'):
            patient.pathway = self.create_patient_pathway(patient)

        # Find current position in pathway
        try:
            current_index = patient.pathway.index(current_dept)
            if current_index + 1 < len(patient.pathway):
                return patient.pathway[current_index + 1]
            else:
                return None  # Patient has completed their pathway
        except ValueError:
            # If current dept not in pathway (shouldn't happen), return to ED
            return "ED"

    def create_patient_pathway(self, patient: Patient) -> List[str]:
        """Create a treatment pathway for a patient"""
        pathway = ["ED"]  # All patients start at ED

        # Critical patients need immediate surgery if required
        if patient.severity == PatientSeverity.CRITICAL and patient.requires_surgery:
            pathway.append("Surgery")
            pathway.append("Recovery")

            # May need specialists after
            if patient.specialist_required:
                pathway.append("Specialists")

            # Possible diagnostics after stabilization
            if patient.requires_imaging:
                pathway.append("Imaging")
            if patient.requires_lab_tests:
                pathway.append("Laboratory")

        else:
            # Non-critical patients typically get diagnostics first
            if patient.requires_imaging:
                pathway.append("Imaging")
            if patient.requires_lab_tests:
                pathway.append("Laboratory")

            # Then specialist if needed
            if patient.specialist_required:
                pathway.append("Specialists")

            # Then surgery if needed
            if patient.requires_surgery:
                pathway.append("Surgery")
                pathway.append("Recovery")

        # Add random chance of returning to diagnostics after specialists
        if "Specialists" in pathway and random.random() < 0.3:
            specialist_index = pathway.index("Specialists")
            if patient.requires_imaging and "Imaging" not in pathway[specialist_index:]:
                pathway.insert(specialist_index + 1, "Imaging")

        # Store pathway with patient for reference
        self.patient_pathways.append((patient.id, pathway))

        return pathway

    def patient_arrived(self, patient: Patient, time: float):
        """Handle a new patient arrival"""
        # All patients start in the Emergency Department
        self.departments["ED"].add_patient_to_queue(patient)

        # Try to serve the patient immediately if resources available
        completion_event = self.departments["ED"].try_serve_next_patient(time)
        if completion_event:
            heapq.heappush(event_queue, completion_event)

    def service_completed(self, event: Event):
        """Handle service completion for a patient"""
        patient = event.entity
        department = event.department
        resource_name = event.resource
        time = event.time

        # Release the resource and complete service
        completed_patient = self.departments[department].complete_service(patient, resource_name, time)

        # Determine next department
        next_dept = self.determine_next_department(patient, department)

        if next_dept:
            # Continue patient journey to next department
            if next_dept in ["Imaging", "Laboratory"]:
                # Handle sub-departments of Diagnostics
                self.departments[next_dept].add_patient_to_queue(patient)
            else:
                self.departments[next_dept].add_patient_to_queue(patient)

            # Try to serve the patient in the new department
            completion_event = self.departments[next_dept].try_serve_next_patient(time)
            if completion_event:
                heapq.heappush(event_queue, completion_event)
        else:
            # Patient has completed treatment
            patient.status = PatientStatus.COMPLETED
            self.completed_patients.append(patient)
            self.statistics["patients_completed"] += 1

            # Update statistics
            wait_time = patient.waiting_time()
            total_time = patient.total_time()

            # Update running averages
            n = len(self.completed_patients)
            self.statistics["average_wait_time"] = ((n - 1) * self.statistics["average_wait_time"] + wait_time) / n
            self.statistics["average_total_time"] = ((n - 1) * self.statistics["average_total_time"] + total_time) / n

        # Try to serve next patient in current department
        next_patient_event = self.departments[department].try_serve_next_patient(time)
        if next_patient_event:
            heapq.heappush(event_queue, next_patient_event)

    def run_simulation(self, duration: float, arrival_rate: float):
        """Run the hospital simulation for specified duration"""
        global current_time

        # Schedule initial patient arrivals
        time = 0
        while time < duration:
            # Next arrival follows exponential distribution
            interarrival_time = random.expovariate(arrival_rate)
            time += interarrival_time

            if time < duration:
                patient = self.generate_patient(time)
                arrival_event = Event(
                    time=time,
                    event_type=EventType.PATIENT_ARRIVAL,
                    department="ED",
                    entity=patient
                )
                heapq.heappush(event_queue, arrival_event)

        # Process events until queue is empty or time exceeds duration
        while event_queue and current_time < duration:
            # Get next event
            event = heapq.heappop(event_queue)

            # Update simulation time
            current_time = event.time

            # Process event
            if event.event_type == EventType.PATIENT_ARRIVAL:
                self.patient_arrived(event.entity, current_time)
            elif event.event_type == EventType.SERVICE_COMPLETION:
                self.service_completed(event)

        # Collect final statistics
        self.collect_statistics()

        return self.statistics

    def collect_statistics(self):
        """Collect statistics from all departments"""
        for name, dept in self.departments.items():
            self.statistics["department_stats"][name] = dept.statistics()

        # Resource utilization
        self.statistics["resource_utilization"] = {
            r.name: r.utilization_time / max(current_time, 1)
            for r in self.resource_allocation.values()
        }

        # Patient pathway analysis
        pathway_counts = defaultdict(int)
        for _, pathway in self.patient_pathways:
            pathway_str = "->".join(pathway)
            pathway_counts[pathway_str] += 1

        self.statistics["common_pathways"] = dict(sorted(
            pathway_counts.items(), key=lambda x: x[1], reverse=True)[:5])


def run_healthcare_simulation():
    """Run the healthcare simulation and return results"""
    # Create hospital
    hospital = Hospital()

    # Run simulation
    simulation_duration = 2880  # 48 hours in minutes
    arrival_rate = 1 / 15  # Average 1 patient every 15 minutes

    print("Starting Virtual Healthcare System Simulation...")
    statistics = hospital.run_simulation(simulation_duration, arrival_rate)

    # Print summary statistics
    print("\nSimulation Complete!")
    print(f"Simulation ran for {simulation_duration} minutes ({simulation_duration / 60:.1f} hours)")
    print(f"Patients generated: {statistics['patients_generated']}")
    print(f"Patients completed: {statistics['patients_completed']}")
    print(f"Average waiting time: {statistics['average_wait_time']:.2f} minutes")
    print(f"Average total time in system: {statistics['average_total_time']:.2f} minutes")

    print("\nDepartment Statistics:")
    for dept, stats in statistics["department_stats"].items():
        if dept in ["Imaging", "Laboratory"]:
            continue  # Skip sub-departments to avoid duplication
        print(f"  {dept}: {stats['patients_processed']} patients processed")

    print("\nMost Common Patient Pathways:")
    for pathway, count in statistics["common_pathways"].items():
        print(f"  {pathway}: {count} patients")

    print("\nResource Utilization (Top 5):")
    top_resources = sorted(statistics["resource_utilization"].items(),
                           key=lambda x: x[1], reverse=True)[:5]
    for resource, utilization in top_resources:
        print(f"  {resource}: {utilization * 100:.1f}%")

    return statistics


if __name__ == "__main__":
    run_healthcare_simulation()
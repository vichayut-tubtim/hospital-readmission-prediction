from pydantic import BaseModel



class PatientData(BaseModel):

    age: str

    gender: str

    time_in_hospital: int

    num_lab_procedures: int

    num_medications: int

    num_procedures: int

    number_inpatient: int

    number_outpatient: int

    number_emergency: int

    number_diagnoses: int
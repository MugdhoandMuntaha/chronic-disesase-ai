-- ==============================================================================
-- EndoPredict AI | Clinical Relational Schema for Supabase PostgreSQL
-- ==============================================================================

-- 1. Patients Master Table
CREATE TABLE IF NOT EXISTS public.patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    ethnicity TEXT NOT NULL,
    smoking_status TEXT NOT NULL,
    baseline_bmi NUMERIC NOT NULL,
    family_history_diabetes INTEGER NOT NULL DEFAULT 0,
    index_date DATE NOT NULL,
    target_label INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 2. Longitudinal Encounters Table
CREATE TABLE IF NOT EXISTS public.encounters (
    encounter_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    encounter_date DATE NOT NULL,
    days_to_index INTEGER NOT NULL,
    encounter_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 3. Vital Signs & Laboratory Measurements Table
CREATE TABLE IF NOT EXISTS public.measurements (
    id BIGSERIAL PRIMARY KEY,
    encounter_id TEXT NOT NULL REFERENCES public.encounters(encounter_id) ON DELETE CASCADE,
    patient_id TEXT NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    measurement_date DATE NOT NULL,
    days_to_index INTEGER NOT NULL,
    systolic_bp NUMERIC,
    diastolic_bp NUMERIC,
    heart_rate NUMERIC,
    bmi NUMERIC,
    fasting_glucose NUMERIC,
    hba1c NUMERIC,
    total_cholesterol NUMERIC,
    ldl NUMERIC,
    hdl NUMERIC,
    triglycerides NUMERIC,
    serum_creatinine NUMERIC,
    egfr NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 4. ICD-10 Comorbidity Diagnoses Table
CREATE TABLE IF NOT EXISTS public.diagnoses (
    diagnosis_id TEXT PRIMARY KEY,
    encounter_id TEXT NOT NULL REFERENCES public.encounters(encounter_id) ON DELETE CASCADE,
    patient_id TEXT NOT NULL REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    diagnosis_date DATE NOT NULL,
    days_to_index INTEGER NOT NULL,
    icd10_code TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 5. Clinical Predictions, Conformal Sets & Recourse Cache Table
CREATE TABLE IF NOT EXISTS public.patient_predictions (
    patient_id TEXT PRIMARY KEY REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    predicted_risk NUMERIC NOT NULL,
    risk_tier TEXT NOT NULL,
    conformal_set JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_ambiguous BOOLEAN NOT NULL DEFAULT false,
    requires_physician_review BOOLEAN NOT NULL DEFAULT false,
    clinical_directive TEXT,
    recourse_actions JSONB DEFAULT '[]'::jsonb,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- ------------------------------------------------------------------------------
-- Performance Indexes
-- ------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_encounters_patient_id ON public.encounters(patient_id);
CREATE INDEX IF NOT EXISTS idx_encounters_date ON public.encounters(encounter_date);
CREATE INDEX IF NOT EXISTS idx_measurements_patient_id ON public.measurements(patient_id);
CREATE INDEX IF NOT EXISTS idx_measurements_encounter_id ON public.measurements(encounter_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_patient_id ON public.diagnoses(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_code ON public.diagnoses(icd10_code);

-- ------------------------------------------------------------------------------
-- Row Level Security (RLS) & Anon Access Policies
-- ------------------------------------------------------------------------------
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.encounters ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.diagnoses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.patient_predictions ENABLE ROW LEVEL SECURITY;

-- Allow public / anon read access (SELECT)
CREATE POLICY "Allow anon read patients" ON public.patients FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow anon read encounters" ON public.encounters FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow anon read measurements" ON public.measurements FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow anon read diagnoses" ON public.diagnoses FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow anon read predictions" ON public.patient_predictions FOR SELECT TO anon, authenticated USING (true);

-- Allow public / anon write access (INSERT, UPDATE)
CREATE POLICY "Allow anon insert patients" ON public.patients FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Allow anon insert encounters" ON public.encounters FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Allow anon insert measurements" ON public.measurements FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Allow anon insert diagnoses" ON public.diagnoses FOR INSERT TO anon, authenticated WITH CHECK (true);
CREATE POLICY "Allow anon upsert predictions" ON public.patient_predictions FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);

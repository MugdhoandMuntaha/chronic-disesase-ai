import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://qnjyvywkkctaocnvszwr.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFuanl2eXdra2N0YW9jbnZzendyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODkyMDcwMTEsImV4cCI6MjEwNDc4MzAxMX0.I7q-4MJHJ0NQtS_1T2zkE2o_ky_aGVCGy5GuI2cuqVw";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

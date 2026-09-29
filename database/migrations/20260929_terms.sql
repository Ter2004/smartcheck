-- Term calendar for the whole-term attendance export. One term applies to all
-- courses; the current term is the one with the latest start_date.
CREATE TABLE IF NOT EXISTS public.terms (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL,                                   -- e.g. 1/2569
    start_date date NOT NULL,                                   -- first day of term
    weeks      int  NOT NULL DEFAULT 16 CHECK (weeks BETWEEN 1 AND 30),
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.terms ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.terms FROM anon, authenticated;
COMMENT ON TABLE public.terms IS 'Academic terms; export columns = schedule weekdays within start_date + weeks.';
NOTIFY pgrst, 'reload schema';

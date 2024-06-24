// Set up indices
CREATE INDEX drug_id_index IF NOT EXISTS FOR (n:Drug) on (n.id);
CREATE INDEX disease_id_index IF NOT EXISTS FOR (n:Disease) on (n.id);
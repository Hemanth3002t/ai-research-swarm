CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS research_tasks (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id)
        ON DELETE CASCADE,

    task_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,

    status VARCHAR(50) NOT NULL DEFAULT 'pending',

    assigned_agent VARCHAR(100),

    result TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);


CREATE TABLE IF NOT EXISTS research_sources (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id)
        ON DELETE CASCADE,

    title TEXT,
    url TEXT,
    content TEXT,

    source_type VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS agent_runs (
    id SERIAL PRIMARY KEY,

    project_id INTEGER REFERENCES projects(id)
        ON DELETE CASCADE,

    agent_name VARCHAR(100) NOT NULL,
    task TEXT,

    model VARCHAR(150),

    status VARCHAR(50) NOT NULL DEFAULT 'started',

    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,

    result TEXT,
    error TEXT,

    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);


CREATE TABLE IF NOT EXISTS drafts (
    id SERIAL PRIMARY KEY,

    project_id INTEGER NOT NULL REFERENCES projects(id)
        ON DELETE CASCADE,

    version INTEGER NOT NULL DEFAULT 1,

    content TEXT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,

    project_id INTEGER NOT NULL REFERENCES projects(id)
        ON DELETE CASCADE,

    draft_id INTEGER REFERENCES drafts(id)
        ON DELETE CASCADE,

    status VARCHAR(50) NOT NULL,

    critical_flaws TEXT,

    revision_notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
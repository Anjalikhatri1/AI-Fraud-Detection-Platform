-- ============================================================
-- Dataverse Schema — AI Fraud Detection Platform
-- Author: Anjali Khatri | O23BCA110060 | Chandigarh University
-- ============================================================

-- TABLE 1: Transactions (cr_transactions)
-- Primary data store for all incoming financial transactions

CREATE TABLE cr_transactions (
    cr_transactionid        UNIQUEIDENTIFIER    PRIMARY KEY,    -- Auto-generated GUID
    cr_userid               NVARCHAR(50)        NOT NULL,       -- USR1234
    cr_amount               DECIMAL(18,2)       NOT NULL,       -- Transaction amount ₹
    cr_paymentmethod        NVARCHAR(50)        NOT NULL,       -- UPI / NEFT / RTGS / IMPS
    cr_device               NVARCHAR(50)        NOT NULL,       -- Mobile / Desktop / Tablet
    cr_state                NVARCHAR(100)       NOT NULL,       -- Indian state
    cr_hourofday            INT                 NOT NULL,       -- 0–23
    cr_transactiondate      DATETIME            NOT NULL,
    cr_isnewpayee           BIT                 DEFAULT 0,
    cr_loginattempts        INT                 DEFAULT 1,
    cr_velocity1h           INT                 DEFAULT 1,      -- Transactions in last 1 hour
    -- AI Scoring Output (populated by Power Automate after API call)
    cr_fraudprobability     DECIMAL(5,4)        DEFAULT NULL,   -- 0.0000 – 1.0000
    cr_riskscore            DECIMAL(5,2)        DEFAULT NULL,   -- 0.00 – 100.00
    cr_riskcategory         NVARCHAR(20)        DEFAULT NULL,   -- Low/Medium/High/Critical
    cr_anomalyflag          BIT                 DEFAULT 0,
    -- Status
    cr_status               NVARCHAR(50)        DEFAULT 'Pending',  -- Pending/Approved/Under Review/Blocked
    cr_isfraud              BIT                 DEFAULT 0,
    cr_createdon            DATETIME            DEFAULT GETUTCDATE(),
    cr_modifiedon           DATETIME            DEFAULT GETUTCDATE()
);

-- Indexes for performance
CREATE INDEX idx_transactions_riskscore   ON cr_transactions (cr_riskscore DESC);
CREATE INDEX idx_transactions_status      ON cr_transactions (cr_status);
CREATE INDEX idx_transactions_date        ON cr_transactions (cr_transactiondate DESC);
CREATE INDEX idx_transactions_userid      ON cr_transactions (cr_userid);


-- TABLE 2: Fraud Cases (cr_fraudcases)
-- Created automatically for High/Critical transactions

CREATE TABLE cr_fraudcases (
    cr_caseid               UNIQUEIDENTIFIER    PRIMARY KEY,
    cr_transactionid        UNIQUEIDENTIFIER    NOT NULL REFERENCES cr_transactions(cr_transactionid),
    cr_riskscore            DECIMAL(5,2)        NOT NULL,
    cr_riskcategory         NVARCHAR(20)        NOT NULL,
    cr_status               NVARCHAR(50)        DEFAULT 'Open',  -- Open/In Progress/Resolved/Closed
    cr_assignedto           NVARCHAR(100),
    cr_investigationnotes   NVARCHAR(MAX),
    cr_resolution           NVARCHAR(100),      -- Confirmed Fraud / False Positive / Inconclusive
    cr_createdon            DATETIME            DEFAULT GETUTCDATE(),
    cr_resolvedon           DATETIME            DEFAULT NULL,
    cr_modifiedon           DATETIME            DEFAULT GETUTCDATE()
);


-- TABLE 3: Alerts (cr_alerts)
-- Notification log for all system alerts

CREATE TABLE cr_alerts (
    cr_alertid              UNIQUEIDENTIFIER    PRIMARY KEY,
    cr_transactionid        UNIQUEIDENTIFIER    REFERENCES cr_transactions(cr_transactionid),
    cr_caseid               UNIQUEIDENTIFIER    REFERENCES cr_fraudcases(cr_caseid),
    cr_alerttype            NVARCHAR(50)        NOT NULL,    -- Email / Push / SMS
    cr_severity             NVARCHAR(20)        NOT NULL,    -- High / Critical
    cr_message              NVARCHAR(500),
    cr_sentto               NVARCHAR(200),
    cr_status               NVARCHAR(20)        DEFAULT 'Sent',   -- Sent / Delivered / Failed
    cr_createdon            DATETIME            DEFAULT GETUTCDATE()
);


-- TABLE 4: Users (cr_users)
-- Platform users — analysts, managers, admins

CREATE TABLE cr_users (
    cr_userid               UNIQUEIDENTIFIER    PRIMARY KEY,
    cr_displayname          NVARCHAR(100)       NOT NULL,
    cr_email                NVARCHAR(200)       NOT NULL UNIQUE,
    cr_role                 NVARCHAR(50)        NOT NULL,    -- Fraud Analyst / Manager / Admin
    cr_isactive             BIT                 DEFAULT 1,
    cr_lastlogin            DATETIME,
    cr_createdon            DATETIME            DEFAULT GETUTCDATE()
);


-- TABLE 5: Risk Configuration (cr_riskconfiguration)
-- Configurable thresholds — editable from Power Apps admin screen

CREATE TABLE cr_riskconfiguration (
    cr_configid             UNIQUEIDENTIFIER    PRIMARY KEY,
    cr_configkey            NVARCHAR(100)       NOT NULL UNIQUE,
    cr_configvalue          NVARCHAR(500)       NOT NULL,
    cr_description          NVARCHAR(500),
    cr_modifiedby           NVARCHAR(100),
    cr_modifiedon           DATETIME            DEFAULT GETUTCDATE()
);

-- Default risk thresholds
INSERT INTO cr_riskconfiguration VALUES
    (NEWID(), 'threshold_critical', '80',  'Risk score >= this value → Critical', 'System', GETUTCDATE()),
    (NEWID(), 'threshold_high',     '60',  'Risk score >= this value → High',     'System', GETUTCDATE()),
    (NEWID(), 'threshold_medium',   '30',  'Risk score >= this value → Medium',   'System', GETUTCDATE()),
    (NEWID(), 'auto_block_score',   '90',  'Auto-block transaction above this score', 'System', GETUTCDATE()),
    (NEWID(), 'alert_email',        'fraud-team@yourbank.com', 'Alert recipient email', 'System', GETUTCDATE());


-- ============================================================
-- Row-Level Security Policies (Dataverse Business Units)
-- ============================================================

-- Fraud Analyst: Read own assigned cases + all Pending/Open transactions
-- Manager:       Read all cases in their business unit
-- Admin:         Full access across all business units

-- Power Automate service account: Create/Update on cr_transactions, cr_fraudcases, cr_alerts
-- Power BI service principal:     Read-only on all tables


-- ============================================================
-- Useful Views for Power BI
-- ============================================================

CREATE VIEW vw_high_risk_transactions AS
SELECT
    t.cr_transactionid,
    t.cr_userid,
    t.cr_amount,
    t.cr_paymentmethod,
    t.cr_device,
    t.cr_state,
    t.cr_riskscore,
    t.cr_riskcategory,
    t.cr_fraudprobability,
    t.cr_status,
    t.cr_transactiondate,
    c.cr_caseid,
    c.cr_status AS case_status
FROM cr_transactions t
LEFT JOIN cr_fraudcases c ON t.cr_transactionid = c.cr_transactionid
WHERE t.cr_riskscore >= 60;


CREATE VIEW vw_daily_fraud_summary AS
SELECT
    CAST(cr_transactiondate AS DATE)    AS txn_date,
    COUNT(*)                            AS total_transactions,
    SUM(CASE WHEN cr_isfraud = 1 THEN 1 ELSE 0 END)      AS fraud_count,
    AVG(cr_amount)                      AS avg_amount,
    AVG(cr_riskscore)                   AS avg_risk_score,
    SUM(CASE WHEN cr_riskcategory = 'Critical' THEN 1 ELSE 0 END) AS critical_count,
    SUM(CASE WHEN cr_riskcategory = 'High'     THEN 1 ELSE 0 END) AS high_count
FROM cr_transactions
GROUP BY CAST(cr_transactiondate AS DATE)
ORDER BY txn_date DESC;

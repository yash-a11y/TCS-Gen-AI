"""Create synthetic Acme Support data and sample policy PDFs."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.config import DB_PATH, POLICY_DIR, ensure_data_dirs
from src.db import get_connection, init_schema
from src.data_loader import ingest_directory

CUSTOMERS = [
    (1, "Ema Patel", "ema.patel@example.com", "+1-415-555-0142", "Business Plus", "active", "2023-04-12", "Jordan Lee"),
    (2, "Marcus Chen", "marcus.chen@example.com", "+1-206-555-0198", "Starter", "active", "2024-01-08", "Jordan Lee"),
    (3, "Sofia Alvarez", "sofia.alvarez@example.com", "+1-512-555-0110", "Enterprise", "active", "2022-11-03", "Priya Shah"),
    (4, "Noah Williams", "noah.williams@example.com", "+1-347-555-0166", "Business Plus", "paused", "2023-09-21", "Priya Shah"),
    (5, "Aisha Khan", "aisha.khan@example.com", "+1-617-555-0133", "Starter", "active", "2025-02-14", "Chris Ng"),
    (6, "Liam O'Connor", "liam.oconnor@example.com", "+1-773-555-0177", "Enterprise", "active", "2021-06-30", "Chris Ng"),
    (7, "Hana Suzuki", "hana.suzuki@example.com", "+1-503-555-0124", "Business Plus", "active", "2024-07-19", "Jordan Lee"),
    (8, "Diego Romero", "diego.romero@example.com", "+1-305-555-0189", "Starter", "churned", "2022-03-11", "Priya Shah"),
    (9, "Emily Brooks", "emily.brooks@example.com", "+1-404-555-0155", "Business Plus", "active", "2023-12-01", "Chris Ng"),
    (10, "Ravi Iyer", "ravi.iyer@example.com", "+1-408-555-0108", "Enterprise", "active", "2020-08-16", "Jordan Lee"),
    (11, "Chloe Martin", "chloe.martin@example.com", "+1-702-555-0144", "Starter", "active", "2025-05-22", "Priya Shah"),
    (12, "Omar Hassan", "omar.hassan@example.com", "+1-214-555-0191", "Business Plus", "active", "2024-03-05", "Chris Ng"),
    (13, "Grace Kim", "grace.kim@example.com", "+1-646-555-0129", "Enterprise", "active", "2022-09-27", "Jordan Lee"),
    (14, "Ben Carter", "ben.carter@example.com", "+1-615-555-0161", "Starter", "paused", "2024-11-18", "Priya Shah"),
    (15, "Nina Rossi", "nina.rossi@example.com", "+1-917-555-0180", "Business Plus", "active", "2023-06-09", "Chris Ng"),
]

TICKETS = [
    (101, 1, "Refund for unused annual seats", "Ema purchased 8 Business Plus seats but 3 teammates left. She wants a prorated refund for unused licenses.", "open", "high", "2026-08-28", None),
    (102, 1, "Invoice shows duplicate charge", "August invoice billed the same add-on twice. Customer asked finance to reverse the duplicate line.", "resolved", "medium", "2026-07-02", "2026-07-04"),
    (103, 1, "SSO login loop after Okta change", "After rotating Okta certificates, Ema could not reach the dashboard. Workaround: password login.", "resolved", "high", "2026-05-16", "2026-05-17"),
    (104, 1, "Export ticket history CSV", "Ema needs last 12 months of support history for an internal audit.", "resolved", "low", "2026-03-09", "2026-03-10"),
    (201, 2, "Cannot reset password", "Reset email never arrives. Checked spam; domain is example.com.", "open", "medium", "2026-09-01", None),
    (202, 3, "SLA credit request", "Two P1 incidents last quarter. Customer wants SLA credit per Enterprise contract.", "open", "high", "2026-08-12", None),
    (203, 3, "Data residency question", "Wants confirmation that EU tickets stay in eu-west-1.", "resolved", "low", "2026-04-22", "2026-04-23"),
    (204, 4, "Pause billing while migrating", "Account paused during ERP cutover. Confirm no charges until they resume.", "open", "medium", "2026-09-04", None),
    (205, 5, "Starter plan feature limit", "Hit 3-agent cap. Asked if refund is possible if they cancel within 14 days.", "resolved", "low", "2026-02-20", "2026-02-21"),
    (206, 6, "Priority routing not firing", "Enterprise priority queue skipped two VIP tickets.", "resolved", "high", "2026-06-11", "2026-06-12"),
    (207, 7, "Mobile app crash on iOS 18", "App closes when attaching screenshots to a ticket.", "open", "medium", "2026-09-10", None),
    (208, 8, "Final invoice after churn", "Customer left in May; still received June invoice.", "resolved", "medium", "2026-06-03", "2026-06-08"),
    (209, 9, "Add account manager to CC", "Wants all P1 emails copied to emily.brooks@example.com.", "resolved", "low", "2026-01-15", "2026-01-15"),
    (210, 10, "Custom retention policy", "Enterprise customer asked for 7-year ticket retention.", "open", "medium", "2026-08-01", None),
    (211, 11, "Onboarding checklist missing", "Welcome email had a broken knowledge-base link.", "resolved", "low", "2026-05-24", "2026-05-25"),
    (212, 12, "Webhook retries failing", "Outbound webhooks return 401 after secret rotation.", "open", "high", "2026-09-08", None),
    (213, 13, "Request SOC 2 report", "Security review for Q4 vendor assessment.", "resolved", "low", "2026-07-18", "2026-07-18"),
    (214, 14, "Downgrade to monthly", "Paused Starter account wants monthly instead of annual.", "open", "low", "2026-09-12", None),
    (215, 15, "Wrong tax on EU invoice", "VAT applied twice on Italian entity.", "resolved", "medium", "2026-04-04", "2026-04-07"),
    (216, 2, "Dark mode contrast", "Sidebar text is hard to read in dark theme.", "resolved", "low", "2026-06-20", "2026-06-21"),
    (217, 7, "API rate limit 429", "Integration bursts exceed 120 req/min on Business Plus.", "open", "medium", "2026-08-30", None),
    (218, 1, "Knowledge base article outdated", "Refund FAQ still mentions 7-day window; Ema thinks policy is 14 days.", "open", "low", "2026-09-14", None),
    (219, 10, "Sandbox environment timeout", "Staging workspace sleeps after 30 minutes of idle time.", "resolved", "medium", "2026-02-11", "2026-02-13"),
    (220, 6, "Add HIPAA BAA", "Legal wants BAA before enabling PHI fields.", "open", "high", "2026-09-02", None),
]

POLICIES = {
    "acme_refund_policy.pdf": {
        "title": "Acme Support — Refund Policy",
        "body": [
            "Effective date: January 1, 2026. This refund policy applies to all Acme Support subscription plans (Starter, Business Plus, and Enterprise).",
            "14-day money-back window: New paid subscriptions may be cancelled for a full refund within 14 calendar days of the original purchase, provided usage is below 1,000 API calls and no Enterprise professional-services hours have been consumed.",
            "Seat reductions: Unused seats on annual Business Plus or Enterprise contracts are eligible for a prorated credit, not a cash refund, when the request is submitted at least 15 days before the next invoice. Credits apply to future invoices only.",
            "Duplicate charges: If billing posts a duplicate charge, Finance will reverse the extra amount within 5 business days after the customer provides the invoice number.",
            "Non-refundable items: Setup fees, custom development, and consumed training workshops are not refundable.",
            "How to request: Open a ticket with subject 'Refund request' or email billing@acme-support.example. Include customer name, plan, and invoice ID. Standard review time is 3 business days.",
            "Chargebacks: Customers should contact Support before filing a chargeback. Unexplained chargebacks may pause the account until resolved.",
        ],
    },
    "acme_privacy_policy.pdf": {
        "title": "Acme Support — Privacy Policy",
        "body": [
            "Acme Support stores customer profile data (name, email, phone, plan) and support ticket contents in order to provide the service.",
            "Ticket attachments may include screenshots. Do not upload government ID numbers, payment card PAN, or health records unless a signed BAA is in place.",
            "Data residency: US customers are hosted in us-east-1. EU Enterprise customers may request eu-west-1 residency.",
            "Retention: Closed tickets are kept for 3 years on Starter and Business Plus, and 7 years on Enterprise when a custom retention add-on is active.",
            "Customers may request an export of their profile and tickets. Exports are delivered as CSV within 2 business days.",
            "We do not sell personal data. Subprocessors are listed in the trust center.",
        ],
    },
    "acme_support_sla.pdf": {
        "title": "Acme Support — Support SLA",
        "body": [
            "Business hours are Monday–Friday 09:00–18:00 in the customer's primary timezone, excluding regional holidays.",
            "Starter: first response in 8 business hours. Business Plus: first response in 4 business hours. Enterprise: first response in 1 hour for P1, 4 hours for P2.",
            "P1 (critical): production outage or login failure affecting all users. P2: major feature broken with a workaround. P3: minor defect or how-to question.",
            "Enterprise customers with two or more confirmed P1 incidents in a quarter may request an SLA service credit of 10% of that quarter's subscription fees.",
            "Priority routing for Enterprise VIP contacts must be enabled by the account manager. If routing fails, the ticket still counts toward SLA once it is in the Enterprise queue.",
            "Scheduled maintenance windows are announced 72 hours in advance and do not count as P1 incidents.",
        ],
    },
}


def write_policy_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 16)]
    for paragraph in paragraphs:
        story.append(Paragraph(paragraph, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)


def seed_database() -> None:
    ensure_data_dirs()
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = get_connection()
    try:
        init_schema(conn)
        conn.executemany(
            """
            INSERT INTO customers (id, name, email, phone, plan, status, signup_date, account_manager)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            CUSTOMERS,
        )
        conn.executemany(
            """
            INSERT INTO tickets (id, customer_id, subject, description, status, priority, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            TICKETS,
        )
        conn.commit()
    finally:
        conn.close()


def seed_policies() -> list[Path]:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, spec in POLICIES.items():
        path = POLICY_DIR / filename
        write_policy_pdf(path, spec["title"], spec["body"])
        written.append(path)
    return written


def main() -> None:
    print("Seeding SQLite customer + ticket data...")
    seed_database()
    conn = get_connection()
    try:
        customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        tickets = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    finally:
        conn.close()
    print(f"  {DB_PATH}: {customers} customers, {tickets} tickets")

    print("Writing sample policy PDFs...")
    paths = seed_policies()
    for path in paths:
        print(f"  {path.name}")

    print("Embedding policies into FAISS...")
    results = ingest_directory(POLICY_DIR)
    for item in results:
        print(f"  {item['source']}: {item['chunks']} chunks")
    print("Seed complete.")


if __name__ == "__main__":
    main()

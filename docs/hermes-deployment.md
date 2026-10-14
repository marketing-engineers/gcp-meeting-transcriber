# Hermes Deployment Guide: Least-Privilege GCP Authentication

This guide describes how to deploy the Transcription Automator to an autonomous agent environment (such as a Hermes agent running on a Linux VPS or home server) with a dedicated, least-privilege Google Cloud identity.

---

## 1. Security Architecture & Threat Model

### Why User ADC is Unsafe for Autonomous Agents
Running `gcloud auth application-default login` on an agent host stores the human user's refresh token on disk with `https://www.googleapis.com/auth/cloud-platform` scope. Because the agent process executes local terminal commands under the host OS user, an agent (or an attacker exploiting prompt injection) could access all Google Cloud resources across all projects accessible to that user account.

**Rule:** The Hermes host OS user must have **no** user-level gcloud login or `~/.config/gcloud` Application Default Credentials.

### The Least-Privilege Model
In `service_account_key` mode, the application uses an isolated Google Cloud Service Account with custom IAM roles restricted strictly to:
1. **Speech API (Project Level):** Submitting batch recognition requests (`speech.recognizers.recognize`), polling operation status (`speech.operations.get`), and service usage quota (`serviceusage.services.use`).
2. **GCS Bucket (Bucket Level Only):** Object operations (`storage.objects.create`, `storage.objects.get`, `storage.objects.list`, `storage.objects.delete`) strictly on a **dedicated Hermes bucket**.

No project-level storage permissions, no administrative roles (`speech.admin`, `storage.admin`), and no service account impersonation permissions (`iam.serviceAccountUser`, `iam.serviceAccountTokenCreator`) are granted.

---

## 2. Administrator GCP Setup Runbook

> **Important:** Run these commands from your **workstation** (where you are authenticated as a project administrator), **never** on the Hermes VPS.

### Step 1: Check Organization Policy
If your project is part of a Google Cloud Organization, verify that service account key creation is permitted:
```bash
export PROJECT_ID="your-project-id"
gcloud resource-manager org-policies describe iam.disableServiceAccountKeyCreation \
    --project=$PROJECT_ID \
    --effective
```
If this policy is enforced (`enforced: true`), an organization policy exception must be granted for this project before continuing.

### Step 2: Create the Service Account
```bash
gcloud iam service-accounts create hermes-transcriber \
    --project=$PROJECT_ID \
    --display-name="Hermes transcriber (least privilege)"
```
Note the resulting service account email (e.g., `hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com`).

### Step 3: Create and Bind Speech Project Role
Create a custom role containing only the three permissions needed for batch transcription and quota attribution:
```bash
gcloud iam roles create hermesTranscriberSpeech \
    --project=$PROJECT_ID \
    --title="Hermes Transcriber Speech" \
    --description="Minimal permissions for Speech-to-Text V2 BatchRecognize" \
    --permissions="speech.recognizers.recognize,speech.operations.get,serviceusage.services.use" \
    --stage="GA"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="projects/${PROJECT_ID}/roles/hermesTranscriberSpeech"
```

### Step 4: Create Dedicated Hermes Bucket
Create a dedicated bucket for Hermes transcription. Do **not** reuse a bucket containing other data.
```bash
export HERMES_BUCKET="${PROJECT_ID}-hermes-transcribe"

gcloud storage buckets create gs://${HERMES_BUCKET} \
    --location=EU \
    --uniform-bucket-level-access \
    --public-access-prevention

# Set a 7-day deletion lifecycle rule for cleanup safety
cat <<EOF > /tmp/lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 7}
    }
  ]
}
EOF
gcloud storage buckets update gs://${HERMES_BUCKET} --lifecycle-file=/tmp/lifecycle.json
rm /tmp/lifecycle.json
```
> **Warning:** Do **not** set a Retention Policy on this bucket. Retention policies prevent object deletion and block the application's automatic post-transcription cleanup.

### Step 5: Bind Bucket Object Permissions
Create a custom role for bucket object operations and bind it **only to the bucket**:
```bash
gcloud iam roles create hermesTranscriberObjects \
    --project=$PROJECT_ID \
    --title="Hermes Transcriber Objects" \
    --description="Object CRUD for temporary transcription audio chunks" \
    --permissions="storage.objects.create,storage.objects.get,storage.objects.list,storage.objects.delete" \
    --stage="GA"

gcloud storage buckets add-iam-policy-binding gs://${HERMES_BUCKET} \
    --member="serviceAccount:hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="projects/${PROJECT_ID}/roles/hermesTranscriberObjects"
```

### Step 6: Speech Service Agent Bucket Access (If Required)
Google Cloud Speech-to-Text V2 may read `gs://` URIs using the Cloud Speech Service Agent. If the first transcription attempt fails reading GCS audio with `PermissionDenied`, grant object viewer rights to the service agent:
```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud storage buckets add-iam-policy-binding gs://${HERMES_BUCKET} \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-speech.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"
```

### Step 7: Generate Service Account Key File
Generate a key file on your workstation:
```bash
gcloud iam service-accounts keys create /tmp/hermes-sa.json \
    --iam-account="hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com" \
    --project=$PROJECT_ID
```
Transfer the key to the Hermes VPS:
```bash
scp /tmp/hermes-sa.json hermes-vps:/home/hermes/.config/gcp-audio-transcribe/hermes-sa.json
rm -f /tmp/hermes-sa.json  # Securely delete local workstation copy
```

### Step 8: Cost & Quota Guards
Because the key is accessible on the VPS filesystem:
1. Set up a **Budget Alert** on your GCP billing account for this project.
2. In Google Cloud Console under **IAM & Admin > Quotas & System Limits**, set a reasonable cap on Cloud Speech-to-Text API request limits (e.g., maximum batch requests per minute/day).

---

## 3. VPS & Hermes Host Configuration

### Host Setup
On the Hermes VPS, run under the unprivileged `hermes` OS user:

1. **Secure the key file:**
   ```bash
   mkdir -p ~/.config/gcp-audio-transcribe
   chmod 700 ~/.config/gcp-audio-transcribe
   chmod 600 ~/.config/gcp-audio-transcribe/hermes-sa.json
   ```

2. **Clone repository and set up virtual environment:**
   ```bash
   git clone <repo-url> ~/marketing-engineers-gcp-audio-transcribe
   cd ~/marketing-engineers-gcp-audio-transcribe
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure `.env` for Hermes:**
   ```bash
   cat <<EOF > .env
   # GCP Authentication Mode
   GCP_AUTH_MODE=service_account_key
   GCP_SERVICE_ACCOUNT_KEY_FILE=/home/hermes/.config/gcp-audio-transcribe/hermes-sa.json
   GCP_EXPECTED_SERVICE_ACCOUNT=hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com

   # GCP Configuration (Note: GCP_PROJECT_ID must be the string project ID)
   GCP_PROJECT_ID=${PROJECT_ID}
   GCS_BUCKET_NAME=${PROJECT_ID}-hermes-transcribe
   GCP_REGION=eu
   TRANSCRIPTION_MODEL=chirp_3
   TRANSCRIPTION_CONCURRENCY=5
   TRANSCRIPTION_MAX_TIMEOUT_SECONDS=1800
   GCS_UPLOAD_TIMEOUT_SECONDS=600
   EOF
   ```

4. **Verify setup with `--check-auth`:**
   ```bash
   venv/bin/python -m src.main --check-auth
   ```
   Expected output:
   ```text
   === GCP Authentication & Permissions Check ===
     Auth Mode: service_account_key
     Principal: hermes-transcriber@...
     Project:   your-project-id
     Bucket:    your-project-id-hermes-transcribe
     Region:    eu
   GCS client initialized successfully.
   Speech-to-Text client initialized successfully...
   GCS bucket '...' IAM permissions verified: all 4 required permissions held.
   GCS bucket '...' list_blobs check succeeded.
   === GCP Authentication Check PASSED ===
   ```

---

## 4. Hermes Agent Configuration & Guardrails

### Installing the Skill
Install the skill into the Hermes skills directory:
```bash
ln -s ~/marketing-engineers-gcp-audio-transcribe/skills/gcp-audio-transcribe ~/.hermes/skills/
```

### Hermes Tool Deny Rules (Soft Guard)
In your Hermes configuration (`~/.hermes/config.yaml`), add soft denial patterns to block routine inspection of authentication tools:
```yaml
approvals:
  deny:
    - "gcloud auth*"
    - "gcloud iam*"
```
*(Note: As documented by Hermes, `approvals.deny` is a policy guardrail, not a cryptographic security boundary. Least privilege at the GCP IAM level is the true enforcement mechanism).*

### Agent Behavior & Delivery Workflow
The skill (`skills/gcp-audio-transcribe/SKILL.md`) directs Hermes to:
1. Deliver the final transcript (`transcripts/<basename>_<lang>_transcript.txt`) as a file attachment rather than dumping text in chat.
2. Deliver the GCP output verbatim without autonomous speaker name inference or rewriting.
3. Prompt the user after delivery before performing any post-processing (e.g., entity resolution, summaries).

---

## 5. Key Rotation & Revocation

### 90-Day Rotation Procedure
1. Create a new key on workstation:
   ```bash
   gcloud iam service-accounts keys create /tmp/new-key.json \
       --iam-account="hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com"
   ```
2. Copy to VPS replacing `~/.config/gcp-audio-transcribe/hermes-sa.json` (`chmod 600`).
3. Run `venv/bin/python -m src.main --check-auth` on the VPS to verify the new key.
4. Delete the old key via `gcloud iam service-accounts keys delete <OLD_KEY_ID>`.

### Emergency Revocation
To immediately revoke access:
```bash
gcloud iam service-accounts disable hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com --project=$PROJECT_ID
```
Or delete all keys:
```bash
gcloud iam service-accounts keys list --iam-account="hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com"
# Delete keys
gcloud iam service-accounts keys delete <KEY_ID> --iam-account="hermes-transcriber@${PROJECT_ID}.iam.gserviceaccount.com"
```

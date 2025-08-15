import os
from google.cloud import dataflow_v1beta3

def cancel_dataflow_jobs(project_id, region):
    """
    Lists and cancels all Dataflow jobs with names starting with 'finetune-'
    that are in a running, pending, or queued state.
    """
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.ListJobsRequest(
        project_id=project_id,
        location=region,
    )

    # The state of the job.
    # JOB_STATE_UNSPECIFIED: The job state is unknown.
    # JOB_STATE_STOPPED: The job is stopped.
    # JOB_STATE_RUNNING: The job is running.
    # JOB_STATE_DONE: The job is done.
    # JOB_STATE_FAILED: The job has failed.
    # JOB_STATE_CANCELLED: The job has been cancelled.
    # JOB_STATE_UPDATED: The job has been updated.
    # JOB_STATE_DRAINING: The job is draining.
    # JOB_STATE_DRAINED: The job has been drained.
    # JOB_STATE_PENDING: The job is pending.
    # JOB_STATE_CANCELLING: The job is cancelling.
    # JOB_STATE_QUEUED: The job is queued.
    # JOB_STATE_RESOURCE_CLEANING_UP: The job is being cleaned up.
    
    terminal_states = [
        dataflow_v1beta3.JobState.JOB_STATE_DONE,
        dataflow_v1beta3.JobState.JOB_STATE_FAILED,
        dataflow_v1beta3.JobState.JOB_STATE_CANCELLED,
        dataflow_v1beta3.JobState.JOB_STATE_STOPPED,
    ]

    page_result = client.list_jobs(request=request)
    
    for job in page_result:
        if job.name.startswith("finetune-") and job.current_state not in terminal_states:
            print(f"Cancelling job {job.name} ({job.id}) in state {job.current_state.name}...")
            
            # Cancel the job
            cancel_request = dataflow_v1beta3.UpdateJobRequest(
                project_id=project_id,
                location=region,
                job_id=job.id,
                job=dataflow_v1beta3.Job(requested_state=dataflow_v1beta3.JobState.JOB_STATE_CANCELLED),
            )
            client.update_job(request=cancel_request)
            print(f"Job {job.name} cancellation request sent.")

if __name__ == "__main__":
    # Ensure you have authenticated with `gcloud auth application-default login`
    project_id = "apache-beam-testing"
    region = "us-central1"
    cancel_dataflow_jobs(project_id, region)

// static/js/waiting.js

// Assumes a global JOB_ID is defined in the HTML template.
var elapsedSeconds = 0;
var timerInterval = setInterval(function(){
  elapsedSeconds++;
  document.getElementById("timer").textContent = "Time elapsed: " + elapsedSeconds + " seconds";
}, 1000);

function checkStatus(){
  fetch('/check_status/' + JOB_ID)
    .then(response => response.json())
    .then(data => {
      if(data.status === "completed"){
        clearInterval(timerInterval);
        window.location.href = "/editor?job_id=" + JOB_ID;
      } else if(data.status === "processing"){
        document.getElementById("queueStatus").textContent = "Your file is currently being processed.";
      } else if(data.status === "pending"){
        document.getElementById("queueStatus").textContent = "Your file is in position " + data.position + " out of " + data.queue_length + " waiting in the queue.";
      } else {
        document.getElementById("queueStatus").textContent = "Status unknown.";
      }
      setTimeout(checkStatus, 2000);
    })
    .catch(error => {
      console.error('Error checking status:', error);
      document.getElementById("queueStatus").textContent = "Error checking status.";
      setTimeout(checkStatus, 2000);
    });
}

checkStatus();

universe = vanilla
Initialdir = /u/gguzman/CS-394N/Final-Project/RAAM
Executable = /lusr/bin/bash
Arguments = /u/gguzman/CS-394N/Final-Project/RAAM/condor/raam_small.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "394N Final Project"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUS = 1
+GPUJob = true
Log =  /u/gguzman/CS-394N/Final-Project/RAAM/logs/condor_python_small.log
Error = /u/gguzman/CS-394N/Final-Project/RAAM/logs/condor_python_small.err  
Output = /u/gguzman/CS-394N/Final-Project/RAAM/logs/condor_python_small.out
Notification = complete
Notify_user = gguzman@cs.utexas.edu
Queue 1

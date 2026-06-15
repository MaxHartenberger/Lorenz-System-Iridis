Interactive Jobs

On occasions, for instance, when needing to use a GUI or for debugging, it can be useful to start jobs directly from a compute node. To do so use the sinteractive command. By default, this will give a single node for 2 hours, but this can be changed with the normal flags to sbatch (see Job Submission). If sufficient resources are available, the interactive job will start immediately, otherwise, it will still need to queue to start and the terminal will be unavailable. Once the job has started the user is logged in to a node of the job and they can run commands from there across all the allocated nodes. By default, the user should also be able to use any GUI interfaces as long as X-forwarding is set up correctly when the user connected to Iridis (by using the -X SSH flag).

As resources may not be available immediately to satisfy the requirements of an interactive job, it is normally only practical to use interactive jobs for short jobs of a few hours or less, running on a handful of nodes. For example, a user may wish to test their applications before submitting a long-running job. Some estimates of what resources are available can be seen with the sinfo command. This will show any idle nodes, along with reserved and allocated nodes.
Using sinteractive

sinteractive is a command line tool that is available on the login nodes.

The most basic usage looks like this:

[username@cyan54 ~]$ sinteractive 

This will start an interactive session on a serial node with 1 CPU and roughly 20 GB of memory.

You can specify the partition to use with sinteractive by doing:

[username@login6002 ~]$ sinteractive --partition=<partitionname> 

Please see our documentation regarding our partitions to select one or run the sinfo command on a login node.

To request more resources like the number of CPUs per task:

[username@cyan54 ~]$ sinteractive --partition=<partitionname>  --cpus-per-task=10

All #SBATCH flags can be passed at the command line to sinteractive which will allow you to customize your sinteractive sessions within the parameters of the settings we have allowed for SLURM.

Common settings that users can apply are to request more time or a custom memory request.

[username@login6001 ~]$ sinteractive --partition=<partitionname>  --time=<custom_time_value> --mem=<custom_memory_value_in_MB>

 

The default sinteractive command in Iridis 5 will assign your interactive job to a gold compute node in the serial partition as a result. By default, sinteractive requests 1 task on 1 node, which by default assigns 1 CPU to that task. Even if you specify a different partition as directed above, you will be given a node in serial unless you ask for more than 20 CPUs.
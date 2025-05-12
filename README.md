# QAOAstudygit
repo for QAOA study the second

setup a workflow as follows:
chosen problem type (mac cut or bin packing) > openQAOAQUBOGenerator.ipynb > qubo_data.json > IBMQQuboSimOptQPUExec.ipynb > job_id > IBMQResultProcessing.ipynb

Issues are that it doesnt seem to give the correct/expected results. The reasons i can think of:
> Formulation of cost hamiltoniain is slightly wrong (can test this by using a perfect simulator)
> initial parameters for the QAOA are poor
> not enough layers are being applied (potentially too many layers if noise is creeping in but theres only 2 so hopefully not or were doomed)
> plotting script has some bug so bitstrings arent displayed properly (can also test this with simulator)
> noise in qunatum circuit is just messing up results (less likely since it doesnt seem to be a completely uniform distribution across bit sterings, but again, can test with simulators)

So i need to test with simulators to see where this issue is coming from

n.b. keeping the older files from previous attempts just incase for now but will delete when i can confirm the workflow prouduces 'correct' (not wrong because of something ive done) results
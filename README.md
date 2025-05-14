# QAOAstudygit
repo for QAOA study the second

setup a workflow as follows:
chosen problem type (mac cut or bin packing) > openQAOAQUBOGenerator.ipynb > qubo_data.json > IBMQQuboSimOptQPUExec.ipynb > job_id > IBMQResultProcessing.ipynb

current issues are that i am not sure i applkying the inital params in the right order, and trying to find the optimal depth of the circuit for performance vs noise accumulation

n.b. keeping the older files from previous attempts just incase for now but will delete when i can confirm the workflow prouduces 'correct' (not wrong because of something ive done) results
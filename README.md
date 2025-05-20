# QAOAstudygit
repo for QAOA study the second

setup a workflow as follows:
chosen problem type (mac cut or bin packing) > openQAOAQUBOGenerator.ipynb > qubo_data.json > IBMQQuboSimOptQPUExec.ipynb > job_id > IBMQResultProcessing.ipynb

current issues are:
1. Uncertain about the pentaly values for the TSp and why the exact hamiltonian energy solver gives me two results
RUN_CONSTRUCTIVE := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json constructive/I$${n}S$${m}.json -seed $$m ; \
		done \
	done

RUN_LAHC := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json heuristic/lahc/I$${n}S$${m}.json -algorithm lahc -seed $$m ; \
		done \
	done

RUN_SA := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json heuristic/sa/I$${n}S$${m}.json -algorithm sa -seed $$m ; \
		done \
	done

RUN_INVERSE_LAHC := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json inverse/sa/I$${n}S$${m}.json -constructive premodel -algorithm lahc -seed $$m ; \
		done \
	done

RUN_INVERSE_SA := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json inverse/sa/I$${n}S$${m}.json -constructive premodel -algorithm sa -seed $$m ; \
		done \
	done

RUN_INVERSE := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json inverse/I$${n}S$${m}.json -constructive premodel -seed $$m ; \
		done \
	done

RUN_FEEDBACK_LAHC := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json feedback/lahc/I$${n}S$${m}.json -feedback 5 -algorithm lahc -seed $$m ; \
		done \
	done

RUN_FEEDBACK_SA := \
	for n in $$(seq 1 10) ; do \
		for m in $$(seq 1 5) ; do \
		python3 src/main.py instance_$$n.json feedback/sa/I$${n}S$${m}.json -feedback 5 -algorithm sa -seed $$m ; \
		done \
	done

RUN_DEFAULT := \
	for n in $$(seq 1 10) ; do \
		python3 src/main.py instance_$$n.json out_$$n.json; \
	done

GEN_INSTANCES := \
	for n in $$(seq 1 10) ; do \
		python3 src/main.py instance_m$$n.json out_m$$n.json ; \
	done

run:
	@$(RUN_CONSTRUCTIVE)

run-lahc:
	@$(RUN_LAHC)

run-sa:
	@$(RUN_SA)

run-inverse:
	@$(RUN_INVERSE)
	@$(RUN_INVERSE_LAHC)
	@$(RUN_INVERSE_SA)

run-feedback:
	@$(RUN_FEEDBACK_LAHC)
	@$(RUN_FEEDBACK_SA)

run-default:
	@$(RUN_DEFAULT)

all: run run-lahc run-sa run-inverse run-feedback
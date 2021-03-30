PYTHON=python3
PATH=lenet5
EPOCHS=3

all: train_export

train_export:
	${PYTHON} -m mnist -o ${OUTPUT} -e ${EPOCHS} --train --export

train:
	${PYTHON} -m mnist -o ${OUTPUT} -e ${EPOCHS} --train

export:
	${PYTHON} -m mnist -o ${OUTPUT} --export
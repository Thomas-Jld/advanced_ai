PYTHON=python3
OUTPUT=lenet5
EPOCHS=3

all: train_export

train_export:
	${PYTHON} -m mnist -o ${OUTPUT} -e ${EPOCHS} --train --save

train:
	${PYTHON} -m mnist -o ${OUTPUT} -e ${EPOCHS} --train

export:
	${PYTHON} -m mnist -o ${OUTPUT} --save
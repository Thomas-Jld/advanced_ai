from torch.optim import AdamW
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from mnist.model import superdupermodel
from mnist.data import MNISTDataset, MNISTLoader

LR = 1e-3
DECAY = 1e-4

EPOCHS = 3

dataset = MNISTDataset()
loader = MNISTLoader()

model = superdupermodel().cuda()
criterion = nn.CrossEntropyLoss().cuda()

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=DECAY)

for epoch in tqdm(range(EPOCHS), desc="Epoch", position=0):
    total_loss = 0.0
    total_acc = 0.0
    
    model = model.train()
    with tqdm(loader.train, "Train", position=1) as pbar:
        for inputs, labels in pbar: 
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, labels)
            acc = (torch.argmax(torch.softmax(out, 1)) == labels).sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() / len(loader.train)
            total_acc += acc.item() / len(dataset.train)
            
            pbar.set_postfix(
                    loss=f"{total_loss:.2e}", 
                    acc=f"{total_acc * 100:.2f}%",
            )

    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0

        model = model.eval()
        with tqdm(loader.test, "Valid", position=2) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.cuda(), labels.cuda()
                
                out = model(inputs)
                loss = criterion(out, labels)
                acc = (torch.argmax(torch.softmax(out, 1)) == labels).sum()

                total_loss += loss.item() / len(loader.test)
                total_acc += acc.item() / len(dataset.test)

                pbar.set_postfix(
                    loss=f"{total_loss:.2e}", 
                    acc=f"{total_acc * 100:.2f}%",
                )

model = model.eval()
trace = torch.jit.trace(
    model.cpu(),
    torch.rand((1, 1, 28, 28)),
)
torch.jit.save(trace, "mnist.pth")

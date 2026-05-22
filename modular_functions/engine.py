
import torch
from typing import Dict, List, Tuple
def train_step(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    device:torch.device
)-> Tuple[float,float]:

    """
    Trains the model for 1 single epoch
    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.to(device)
    model.train()
    train_loss,train_acc = 0,0

    for batch,(x,y) in enumerate(dataloader):
      x,y = x.to(device) , y.to(device)
      y_logits = model(x)

      loss = loss_fn(y_logits,y)
      train_loss = train_loss + loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y_pred = torch.argmax(torch.softmax(y_logits,dim=1),dim=1)
      train_acc = train_acc + ((torch.eq(y,y_pred).sum().item())/len(y))*100

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    device:torch.device
)-> Tuple[float,float]:

    """
    Tests the model for 1 single epoch
    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to minimize.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.to(device)
    model.eval()
    test_loss,test_acc = 0,0

    with torch.inference_mode():
          for batch,(x,y) in enumerate(dataloader):
            x,y = x.to(device) , y.to(device)
            y_logits = model(x)

            loss = loss_fn(y_logits,y)
            test_loss = test_loss + loss.item()

            y_pred = torch.argmax(torch.softmax(y_logits,dim=1),dim=1)
            test_acc = test_acc + ((torch.eq(y,y_pred).sum().item())/len(y))*100

          test_loss = test_loss / len(dataloader)
          test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

          """
              Args:
              model: A PyTorch model to be trained and tested.
              train_dataloader: A DataLoader instance for the model to be trained on.
              test_dataloader: A DataLoader instance for the model to be tested on.
              optimizer: A PyTorch optimizer to help minimize the loss function.
              loss_fn: A PyTorch loss function to calculate loss on both datasets.
              epochs: An integer indicating how many epochs to train for.
              device: A target device to compute on (e.g. "cuda" or "cpu").

              Returns:
              A dictionary of training and testing loss as well as training and
              testing accuracy metrics. Each metric has a value in a list for 
              each epoch.
              In the form: {train_loss: [...],
                            train_acc: [...],
                            test_loss: [...],
                            test_acc: [...]} 
              For example if training for epochs=2: 
                          {train_loss: [2.0616, 1.0537],
                            train_acc: [0.3945, 0.3945],
                            test_loss: [1.2641, 1.5706],
                            test_acc: [0.3400, 0.2973]} 
              """
          results = {
              "train_loss":[],
              "train_acc":[],
              "test_loss":[],
              "test_acc":[]
          }

          for epoch in range(epochs):
            train_loss,train_acc = train_step(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device
            )
            test_loss,test_acc = test_step(
                model,
                test_dataloader,
                loss_fn,
                device
            )

            print(
                  f"Epoch: {epoch+1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_acc: {test_acc:.4f}"
             )
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

          return results

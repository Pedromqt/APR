from sklearn.metrics import confusion_matrix, f1_score
import torch

def fit(train_data, model, criterion, optimizer, n_epochs, to_device=True, flatten=False , use_nll=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if to_device:
        model = model.to(device)

    acc_values  = []
    loss_values = []
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        acc_sum = 0
        for X_batch, y_batch in train_data:
            if flatten:
                X_batch = X_batch.view(X_batch.size(0), -1)  # flatten the input
            X_batch = X_batch.to(device)
            y_batch = y_batch.view(-1).long().to(device)

            output = model(X_batch, use_nll=use_nll)             # forward pass
            loss = criterion(output, y_batch)                    # compute loss

            optimizer.zero_grad()                                # clear gradients
            loss.backward()                                      # backpropagation
            optimizer.step()                                     # update weights

            epoch_loss += loss.item()
            
            _, preds  = torch.max(output, 1)
            batch_acc = (preds == y_batch).sum().item() / y_batch.size(0)
            acc_sum += batch_acc
            
        avg_loss = epoch_loss / len(train_data)
        avg_acc  = acc_sum  / len(train_data)
        acc_values.append(avg_acc*100)
        loss_values.append(avg_loss)
    
    return model.to("cpu"), loss_values, acc_values

def evaluate_nn(nn, loader, flatten=False,file = None,use_nll=False):
    nn.eval()
    all_preds = []
    all_labels = []
    acc_values  = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn = nn.to(device)
    total_samples = 0
    total_acc = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            if flatten:
                X_batch = X_batch.view(X_batch.size(0), -1)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
                
            output = nn(X_batch,use_nll=use_nll)
            _, predicted = torch.max(output, 1)
            batch_acc = (predicted == y_batch).sum().item() 
            total_acc += batch_acc
            total_samples += y_batch.size(0)
            all_preds.append(predicted.cpu()    )
            all_labels.append(y_batch.cpu())

    acc = total_acc / total_samples
    acc_values.append(acc*100)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    conf_mat = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if file is not None:
        file.write('Confusion Matrix:\n')
        file.write(str(conf_mat) + '\n')
        file.write('F1 Score: ' + str(f1) + '\n')
    return acc_values, conf_mat, f1
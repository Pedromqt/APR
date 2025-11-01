from sklearn.metrics import confusion_matrix, f1_score
import torch

def fit(train_data, model, criterion, optimizer, n_epochs, to_device=True, flatten=False , use_nll=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if to_device:
        model = model.to(device)

    loss_values = []
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_data:
            if flatten: # DNN
                X_batch = X_batch.view(X_batch.size(0), -1)  # flatten the input
            X_batch = X_batch.to(device)
            y_batch = y_batch.view(-1).long().to(device)

            output = model(X_batch, use_nll=use_nll)             # forward pass
            loss = criterion(output, y_batch)   # compute loss

            optimizer.zero_grad()               # clear gradients
            loss.backward()                     # backpropagation
            optimizer.step()                    # update weights

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        loss_values.append(avg_loss)
    
    return model.to("cpu")

def evaluate_nn(nn, loader, flatten=False,file = None,use_nll=False):
    nn.eval()
    all_preds = []
    all_labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn = nn.to(device)
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            if flatten: # DNN
                X_batch = X_batch.view(X_batch.size(0), -1)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
                
            output = nn(X_batch,use_nll=use_nll)
            _, predicted = torch.max(output, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    conf_mat = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if file is not None:
        file.write('Confusion Matrix:\n')
        file.write(str(conf_mat) + '\n')
        file.write('F1 Score: ' + str(f1) + '\n')
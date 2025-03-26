import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.randn((200, 300, 200, 20), device=device)
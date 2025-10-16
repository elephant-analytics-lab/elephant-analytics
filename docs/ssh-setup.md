# SSH Setup (ed25519)

1) ssh-keygen -t ed25519 -C "you@example.com"
2) ssh-agent -s; ssh-add ~/.ssh/id_ed25519
3) Add id_ed25519.pub to GitHub → SSH Keys
4) ssh -T git@github.com

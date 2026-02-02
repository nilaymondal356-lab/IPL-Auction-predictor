# AWS Deployment Guide for IPL Auction Predictor

## Prerequisites
- AWS Account
- Basic knowledge of AWS EC2, Security Groups
- SSH client installed on your local machine
- Domain name (optional, for production)

## Deployment Options

### Option 1: AWS EC2 (Recommended for Production)

#### Step 1: Launch EC2 Instance

1. **Login to AWS Console**
   - Go to AWS Console → EC2 → Launch Instance

2. **Configure Instance**
   - **Name**: ipl-auction-predictor
   - **AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance Type**: t2.micro (Free tier) or t2.small (Better performance)
   - **Key Pair**: Create new or use existing (save .pem file)
   - **Network Settings**:
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere
     - Allow HTTPS (port 443) from anywhere
     - Allow Custom TCP (port 5000) from anywhere (for testing)

3. **Storage**: 15 GB (default is fine)

4. **Launch Instance**

#### Step 2: Connect to EC2 Instance

```bash
# Windows (using PowerShell or Git Bash)
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip

# macOS/Linux
chmod 400 your-key.pem
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

#### Step 3: Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs -y

# Install git (if needed)
sudo apt install git -y

# Verify installations
python3 --version
node --version
npm --version
```

#### Step 4: Deploy Application

```bash
# Create directory
mkdir -p ~/apps
cd ~/apps

# Upload your project (Option A: Using SCP)
# From your local machine:
scp -i "your-key.pem" -r ipl-auction-predictor ubuntu@your-ec2-ip:~/apps/

# OR (Option B: Using Git)
git clone your-repository-url
cd ipl-auction-predictor

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build frontend
cd frontend
npm install
npm run build
cd ..

# Train model if not included
python model.py
```

#### Step 5: Install and Configure Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Test Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Press Ctrl+C to stop
```

#### Step 6: Create Systemd Service (Auto-start on boot)

```bash
# Create service file
sudo nano /etc/systemd/system/ipl-predictor.service
```

Add the following content:
```ini
[Unit]
Description=IPL Auction Predictor
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/apps/ipl-auction-predictor
Environment="PATH=/home/ubuntu/apps/ipl-auction-predictor/venv/bin"
ExecStart=/home/ubuntu/apps/ipl-auction-predictor/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Reload systemd, enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ipl-predictor
sudo systemctl start ipl-predictor

# Check status
sudo systemctl status ipl-predictor

# View logs
sudo journalctl -u ipl-predictor -f
```

#### Step 7: Install and Configure Nginx (Production)

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/ipl-predictor
```

Add the following content:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or your EC2 public IP

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ipl-predictor /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Enable Nginx to start on boot
sudo systemctl enable nginx
```

#### Step 8: Configure Firewall (UFW)

```bash
# Enable UFW
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Check status
sudo ufw status
```

#### Step 9: Setup SSL Certificate (Optional, for HTTPS)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured by default
# Test renewal
sudo certbot renew --dry-run
```

#### Step 10: Access Your Application

- **Without domain**: `http://your-ec2-public-ip`
- **With domain**: `http://your-domain.com`
- **With SSL**: `https://your-domain.com`

---

### Option 2: AWS Elastic Beanstalk (Simpler Deployment)

#### Step 1: Install EB CLI

```bash
# Windows
pip install awsebcli

# macOS
brew install awsebcli

# Linux
pip install awsebcli --user
```

#### Step 2: Initialize Elastic Beanstalk

```bash
cd ipl-auction-predictor

# Initialize EB
eb init -p python-3.8 ipl-auction-predictor --region us-east-1

# Create environment
eb create ipl-predictor-env
```

#### Step 3: Deploy

```bash
eb deploy
```

#### Step 4: Open Application

```bash
eb open
```

---

### Option 3: AWS Lightsail (Budget-Friendly)

1. **Create Lightsail Instance**
   - Go to AWS Lightsail Console
   - Create Instance
   - Choose OS Only → Ubuntu 22.04
   - Choose plan ($3.50/month or $5/month)
   - Launch Instance

2. **Connect and Deploy**
   - Use SSH from browser console
   - Follow Steps 3-9 from EC2 deployment

---

## Monitoring and Maintenance

### Check Application Status
```bash
# Service status
sudo systemctl status ipl-predictor

# View logs
sudo journalctl -u ipl-predictor -n 50

# Nginx status
sudo systemctl status nginx

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Restart Services
```bash
# Restart application
sudo systemctl restart ipl-predictor

# Restart Nginx
sudo systemctl restart nginx
```

### Update Application
```bash
cd ~/apps/ipl-auction-predictor

# Pull updates (if using git)
git pull

# Rebuild frontend
cd frontend
npm run build
cd ..

# Restart service
sudo systemctl restart ipl-predictor
```

---

## Performance Optimization

### 1. Increase Gunicorn Workers
```bash
# Edit service file
sudo nano /etc/systemd/system/ipl-predictor.service

# Change workers based on CPU cores: (2 × CPU cores) + 1
ExecStart=... gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ipl-predictor
```

### 2. Enable Nginx Caching
```nginx
# Add to Nginx config
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m inactive=60m;
proxy_cache my_cache;
```

### 3. Enable Gzip Compression
```nginx
# Add to Nginx config
gzip on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
```

---

## Security Best Practices

1. **Keep System Updated**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Use Environment Variables for Secrets**
   ```bash
   # Create .env file
   nano /home/ubuntu/apps/ipl-auction-predictor/.env
   
   # Add secrets
   SECRET_KEY=your-secret-key
   
   # Update service file to load .env
   ```

3. **Limit SSH Access**
   - Use key-based authentication only
   - Disable password login
   - Use fail2ban for brute force protection

4. **Regular Backups**
   ```bash
   # Backup data
   tar -czf backup-$(date +%Y%m%d).tar.gz ~/apps/ipl-auction-predictor
   ```

---

## Troubleshooting

### Application Not Starting
```bash
# Check logs
sudo journalctl -u ipl-predictor -n 100

# Check Python environment
source venv/bin/activate
python app.py  # Run directly to see errors
```

### 502 Bad Gateway
```bash
# Check if Gunicorn is running
ps aux | grep gunicorn

# Restart service
sudo systemctl restart ipl-predictor
```

### Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

---

## Cost Estimation

### EC2 t2.micro (Free Tier)
- **First 12 months**: Free
- **After**: ~$8-10/month

### EC2 t2.small
- **Cost**: ~$17/month

### Lightsail
- **512 MB RAM**: $3.50/month
- **1 GB RAM**: $5/month
- **2 GB RAM**: $10/month

### Domain Name (Optional)
- **Route 53**: ~$12/year
- **Third-party**: $10-15/year

---

## Support

For issues during deployment:
1. Check application logs
2. Check Nginx logs
3. Verify security group settings
4. Check systemd service status
5. Review this guide carefully

**Happy Deploying! 🚀**

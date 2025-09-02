# उम्मीद से हरी | Ummid Se Hari

**Smart & Carbon-Free Village PWA for Damday Gram Panchayat**

A comprehensive digital governance platform built with Next.js 14, featuring advanced admin capabilities, real-time data management, and offline-first architecture.

## 🚀 Features

### 🏛️ Digital Governance
- **Citizen Services Portal** - Online applications, complaint management, service requests
- **Project Transparency** - Real-time budget tracking, milestone updates, contractor management
- **Scheme Management** - Automated eligibility checking, application processing
- **Event & Notice Management** - Community announcements, calendar integration

### 🌱 Carbon-Free Mission
- **Carbon Footprint Calculator** - Household emission tracking with actionable recommendations
- **Solar Adoption Wizard** - ROI calculations, installation guidance
- **Tree Pledge Wall** - Community tree planting commitments with verification
- **Waste Management Game** - Interactive education on waste segregation

### 🔐 Advanced Security
- **Multi-factor Authentication** - Email OTP + TOTP 2FA for administrators
- **Role-based Access Control** - Granular permissions (Admin, Approver, Editor, Data Entry, Viewer)
- **Comprehensive Audit Logging** - All actions tracked with IP, user agent, and detailed changes
- **Session Security** - Secure JWT tokens with automatic renewal

### 📱 Progressive Web App
- **Offline-first Architecture** - Works without internet, syncs when connected
- **Background Sync** - Queued form submissions for offline scenarios  
- **App-like Experience** - Install on mobile devices, push notifications
- **Responsive Design** - Optimized for mobile, tablet, and desktop

### 🌍 Internationalization
- **Bilingual Interface** - Hindi (primary) and English support
- **Dynamic Language Switching** - User preference persistence
- **Content Management** - Multilingual content blocks and pages
- **RTL Support Ready** - Prepared for additional languages

## 🏗️ Architecture

### Frontend
- **Next.js 14** - App Router, React Server Components
- **TypeScript** - Full type safety throughout
- **Tailwind CSS + shadcn/ui** - Modern, accessible component library
- **Framer Motion** - Smooth animations with accessibility considerations

### Backend  
- **Next.js API Routes** - Serverless functions with middleware
- **tRPC** - End-to-end type-safe APIs (planned for advanced features)
- **Prisma ORM** - Type-safe database operations
- **PostgreSQL** - Production-ready relational database

### Security & Authentication
- **NextAuth.js** - Industry-standard authentication
- **TOTP 2FA** - Time-based one-time passwords with backup codes
- **RBAC** - Fine-grained role and permission management
- **Audit Trails** - Complete action logging and monitoring

### Infrastructure
- **Docker Compose** - Local development environment
- **Vercel Deployment** - Production hosting with edge functions
- **MinIO/S3** - Object storage for files and media
- **Redis** - Caching and session management

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/ummid-se-hari.git
   cd ummid-se-hari
   ```

2. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Quick setup with script**
   ```bash
   npm run setup
   ```

   Or manual setup:
   ```bash
   # Start services
   docker-compose up -d
   
   # Install dependencies
   npm install
   
   # Setup database
   npm run db:migrate
   npm run db:seed
   
   # Start development server
   npm run dev
   ```

4. **Access the application**
   - **Frontend**: http://localhost:3000
   - **Admin Panel**: http://localhost:3000/admin
   - **Admin Login**: admin@ummidsehari.in (use email OTP)

## 🛠️ Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Production build  
npm run start        # Production server
npm run typecheck    # TypeScript validation
npm run lint         # ESLint checking
npm run test         # Run unit tests
npm run e2e          # End-to-end tests

# Database
npm run db:migrate   # Run migrations
npm run db:seed      # Seed sample data
npm run db:studio    # Open Prisma Studio
npm run db:reset     # Reset database

# Backup & Restore
npm run backup       # Create database backup
npm run restore      # Restore from backup
```

### Database Management

**Create Migration:**
```bash
npx prisma migrate dev --name your-migration-name
```

**Reset Database:**
```bash
npm run db:reset
npm run db:seed
```

**Backup Database:**
```bash
npm run backup                    # Full SQL backup
npm run backup -- --json         # JSON format
npm run backup -- --schema-only  # Schema only
```

## 🔐 Admin Features

### User Management
- Create, edit, delete users with role assignments
- Bulk operations for multiple users
- Advanced filtering and search
- Activity monitoring and session tracking

### Audit Logging  
- Real-time activity tracking
- Advanced filtering by user, action, entity, date
- Detailed change logs with before/after states
- IP address and user agent tracking

### 2FA Management
- QR code generation for TOTP setup
- Backup code generation and management
- Force 2FA for administrative roles
- Audit trail for security events

### Content Management
- Block-based page editor (planned)
- Media library with alt text requirements  
- Version control and publishing workflow
- SEO optimization tools

## 🌍 Deployment

### Vercel (Recommended)

1. **Connect to Vercel**
   ```bash
   npm install -g vercel
   vercel login
   vercel
   ```

2. **Configure Environment Variables**
   - Add all variables from `.env.example`
   - Setup PostgreSQL database (Neon, Supabase, or Railway)
   - Configure S3-compatible storage

3. **Deploy**
   ```bash
   vercel --prod
   ```

### Docker Production

1. **Build production image**
   ```bash
   docker build -t ummid-se-hari .
   ```

2. **Run with docker-compose**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL="postgresql://user:pass@localhost:5432/ummid_se_hari"

# NextAuth
NEXTAUTH_URL="http://localhost:3000"
NEXTAUTH_SECRET="your-secret-key"

# Email (Gmail example)
EMAIL_SERVER_HOST="smtp.gmail.com"
EMAIL_SERVER_PORT=587
EMAIL_SERVER_USER="your-email@gmail.com"
EMAIL_SERVER_PASSWORD="your-app-password"

# Storage (MinIO/S3)
S3_ENDPOINT="http://localhost:9000"
S3_ACCESS_KEY_ID="minioadmin"
S3_SECRET_ACCESS_KEY="minioadmin"
```

## 🧪 Testing

### Unit Tests
```bash
npm test                 # Run all tests
npm run test:watch       # Watch mode
npm run test:coverage    # Coverage report
```

### E2E Tests
```bash
npm run e2e              # Run Playwright tests
npm run e2e:ui           # Interactive mode
```

### Manual Testing
- Test offline functionality by disabling network
- Verify 2FA setup with authenticator apps
- Test file uploads and media management
- Check responsive design on mobile devices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Standards
- TypeScript strict mode
- ESLint configuration enforced
- Prettier for code formatting
- Conventional commit messages
- Component testing required

## 📚 Documentation

- [Admin Guide](docs/ADMIN_GUIDE.md) - Complete admin interface guide
- [API Documentation](docs/API.md) - REST API endpoints
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Contributing Guide](docs/CONTRIBUTING.md) - Development guidelines

## 🗓️ Roadmap

### Phase 1 ✅ (Completed)
- [x] Bootstrap & DX setup
- [x] Database & Authentication
- [x] PWA & Service Worker
- [x] Design System
- [x] Admin Shell

### Phase 2 🚧 (In Progress) 
- [ ] Content Management System
- [ ] Form Builder & Submissions
- [ ] GIS & Mapping Integration
- [ ] Project & Budget Management

### Phase 3 📋 (Planned)
- [ ] Smart Mission Tools
- [ ] Scheme & Eligibility Engine
- [ ] Services & Request Management
- [ ] News/Notices/Events

### Phase 4 🔮 (Future)
- [ ] Advanced Analytics
- [ ] Multi-tenant Support
- [ ] Mobile Apps
- [ ] AI-powered Insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Damday Gram Panchayat** - For the vision of digital governance
- **Next.js Team** - For the excellent framework
- **Prisma Team** - For the outstanding ORM
- **Shadcn** - For the beautiful UI components
- **Open Source Community** - For the tools and inspiration

---

**Built with ❤️ for Rural India's Digital Future**

*"उम्मीद से हरी" - Green with Hope*

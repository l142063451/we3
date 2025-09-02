#!/bin/bash

# Database setup script for Ummid Se Hari
# This script handles the complete database setup and seeding

set -e

echo "🌱 Setting up Ummid Se Hari database..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start database services
echo "🐳 Starting database services..."
docker-compose up -d postgres redis minio

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U postgres; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Generate Prisma client (requires network access)
echo "🔧 Generating Prisma client..."
if command -v npx > /dev/null; then
    npx prisma generate || {
        echo "⚠️  Could not generate Prisma client (network issue)"
        echo "   You'll need to run 'npx prisma generate' manually with internet access"
    }
else
    echo "⚠️  npx not found, skipping Prisma client generation"
fi

# Run database migrations
echo "🔄 Running database migrations..."
if npx prisma migrate dev --name init; then
    echo "✅ Database migrations completed!"
else
    echo "⚠️  Migrations failed, you may need to run them manually"
fi

# Seed the database
echo "🌱 Seeding database with sample data..."
if npm run db:seed; then
    echo "✅ Database seeded successfully!"
    echo ""
    echo "📋 Admin User Created:"
    echo "   Email: admin@ummidsehari.in"
    echo "   Password: Use email OTP login"
    echo "   2FA: Enabled (check seed output for QR code URL)"
    echo ""
else
    echo "⚠️  Seeding failed, you may need to run 'npm run db:seed' manually"
fi

echo "🎉 Database setup completed!"
echo ""
echo "🚀 Next steps:"
echo "   1. Copy .env.example to .env and configure your settings"
echo "   2. Run 'npm run dev' to start the development server"
echo "   3. Visit http://localhost:3000/admin to access the admin panel"
echo ""
echo "🔗 Services running:"
echo "   - PostgreSQL: localhost:5432"
echo "   - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "   - Mailhog: http://localhost:8025"
echo "   - Redis: localhost:6379"
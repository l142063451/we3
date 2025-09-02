import { PrismaClient, UserRole } from '@prisma/client';
import * as bcrypt from 'bcryptjs';
import * as speakeasy from 'speakeasy';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seeding...');

  // Create admin user with TOTP 2FA
  const adminSecret = speakeasy.generateSecret({ name: 'Ummid Se Hari Admin' });
  
  const adminUser = await prisma.user.upsert({
    where: { email: 'admin@ummidsehari.in' },
    update: {},
    create: {
      email: 'admin@ummidsehari.in',
      name: 'पंचायत व्यवस्थापक',
      phone: '+91-9876543210',
      roles: [UserRole.ADMIN],
      locale: 'hi-IN',
      twoFAEnabled: true,
      twoFASecret: adminSecret.base32,
      emailVerified: new Date(),
    },
  });

  console.log('✅ Admin user created:', adminUser.email);

  // Create sample roles users
  const editorUser = await prisma.user.upsert({
    where: { email: 'editor@ummidsehari.in' },
    update: {},
    create: {
      email: 'editor@ummidsehari.in',
      name: 'संपादक',
      roles: [UserRole.EDITOR],
      locale: 'hi-IN',
      emailVerified: new Date(),
    },
  });

  const dataEntryUser = await prisma.user.upsert({
    where: { email: 'dataentry@ummidsehari.in' },
    update: {},
    create: {
      email: 'dataentry@ummidsehari.in',
      name: 'डेटा एंट्री ऑपरेटर',
      roles: [UserRole.DATA_ENTRY],
      locale: 'hi-IN',
      emailVerified: new Date(),
    },
  });

  console.log('✅ Staff users created');

  // Create sample pages with blocks
  const homePage = await prisma.page.upsert({
    where: { slug: 'home' },
    update: {},
    create: {
      slug: 'home',
      title: {
        hi: 'उम्मीद से हरी - दामदय ग्राम पंचायत',
        en: 'Ummid Se Hari - Damday Gram Panchayat'
      },
      locale: 'hi-IN',
      status: 'PUBLISHED',
      blocks: [
        {
          type: 'hero',
          id: 'hero-1',
          content: {
            title: {
              hi: 'स्वागत है उम्मीद से हरी में',
              en: 'Welcome to Ummid Se Hari'
            },
            subtitle: {
              hi: 'चुआनाला दामदय ग्राम पंचायत - स्मार्ट और कार्बन-मुक्त गाँव',
              en: 'Chuanala Damday Gram Panchayat - Smart & Carbon-Free Village'
            },
            image: '/images/hero-damday.jpg',
            cta: {
              text: {
                hi: 'सेवाएं देखें',
                en: 'View Services'
              },
              href: '/services'
            }
          }
        },
        {
          type: 'announcement',
          id: 'announcement-1',
          content: {
            title: {
              hi: '🚨 महत्वपूर्ण सूचना',
              en: '🚨 Important Notice'
            },
            message: {
              hi: 'नया वार्ड सदस्य चुनाव 15 दिसंबर 2024 को होगा',
              en: 'New ward member election will be held on December 15, 2024'
            },
            type: 'info',
            dismissible: true
          }
        },
        {
          type: 'stats',
          id: 'stats-1',
          content: {
            items: [
              {
                label: { hi: 'कुल परियोजनाएं', en: 'Total Projects' },
                value: '24',
                trend: '+3'
              },
              {
                label: { hi: 'लगाए गए पेड़', en: 'Trees Planted' },
                value: '1,250',
                trend: '+125'
              },
              {
                label: { hi: 'सोलर पैनल (kW)', en: 'Solar Panels (kW)' },
                value: '85',
                trend: '+12'
              },
              {
                label: { hi: 'समाधान की दर', en: 'Resolution Rate' },
                value: '94%',
                trend: '+2%'
              }
            ]
          }
        }
      ],
      seo: {
        title: {
          hi: 'उम्मीद से हरी - दामदय ग्राम पंचायत',
          en: 'Ummid Se Hari - Damday Gram Panchayat'
        },
        description: {
          hi: 'चुआनाला दामदय ग्राम पंचायत का आधिकारिक पोर्टल। सेवाएं, परियोजनाएं, और नागरिक सुविधाएं।',
          en: 'Official portal of Chuanala Damday Gram Panchayat. Services, projects, and citizen facilities.'
        }
      },
      publishedAt: new Date(),
      createdById: adminUser.id,
      updatedById: adminUser.id,
    },
  });

  console.log('✅ Home page created');

  // Create sample projects
  const projects = await Promise.all([
    prisma.project.create({
      data: {
        title: {
          hi: 'सोलर स्ट्रीट लाइट प्रोजेक्ट',
          en: 'Solar Street Light Project'
        },
        description: {
          hi: 'गांव की मुख्य सड़कों पर सोलर स्ट्रीट लाइट लगाना',
          en: 'Installing solar street lights on main village roads'
        },
        type: 'infrastructure',
        ward: 'वार्ड-1',
        budget: 500000,
        spent: 375000,
        status: 'IN_PROGRESS',
        startDate: new Date('2024-01-15'),
        endDate: new Date('2024-12-31'),
        geo: {
          type: 'Point',
          coordinates: [77.2090, 28.6139]
        },
        tags: ['solar', 'infrastructure', 'lighting'],
        milestones: {
          create: [
            {
              title: {
                hi: 'साइट सर्वे पूरा',
                en: 'Site Survey Complete'
              },
              description: {
                hi: 'सभी लोकेशन का सर्वे और मार्किंग',
                en: 'Survey and marking of all locations'
              },
              targetDate: new Date('2024-02-28'),
              actualDate: new Date('2024-02-25'),
              progress: 100,
              photos: ['/images/survey-1.jpg', '/images/survey-2.jpg']
            },
            {
              title: {
                hi: 'लाइट इंस्टॉलेशन',
                en: 'Light Installation'
              },
              targetDate: new Date('2024-06-30'),
              progress: 75,
            }
          ]
        }
      }
    }),
    
    prisma.project.create({
      data: {
        title: {
          hi: 'वृक्षारोपण अभियान 2024',
          en: 'Tree Plantation Drive 2024'
        },
        description: {
          hi: 'गांव में 2000 पेड़ लगाने का अभियान',
          en: 'Campaign to plant 2000 trees in the village'
        },
        type: 'environment',
        budget: 150000,
        spent: 85000,
        status: 'IN_PROGRESS',
        startDate: new Date('2024-07-01'),
        endDate: new Date('2025-03-31'),
        tags: ['trees', 'environment', 'carbon-neutral']
      }
    })
  ]);

  console.log('✅ Sample projects created');

  // Create sample schemes
  const schemes = await Promise.all([
    prisma.scheme.create({
      data: {
        title: {
          hi: 'प्रधानमंत्री आवास योजना - ग्रामीण',
          en: 'Pradhan Mantri Awas Yojana - Rural'
        },
        description: {
          hi: 'ग्रामीण क्षेत्र में पक्का मकान बनाने के लिए सहायता योजना',
          en: 'Assistance scheme for building pucca houses in rural areas'
        },
        category: 'housing',
        criteriaJSON: {
          eligibility: [
            { field: 'annualIncome', operator: '<', value: 200000 },
            { field: 'hasLand', operator: '==', value: true },
            { field: 'currentHouseType', operator: '==', value: 'kutcha' }
          ],
          documents: ['aadhar', 'income_certificate', 'land_documents', 'bank_passbook']
        },
        docsRequired: ['आधार कार्ड', 'आय प्रमाण पत्र', 'भूमि दस्तावेज', 'बैंक पासबुक'],
        processSteps: {
          steps: [
            {
              title: { hi: 'आवेदन भरें', en: 'Fill Application' },
              description: { hi: 'ऑनलाइन या ऑफलाइन आवेदन करें', en: 'Apply online or offline' }
            },
            {
              title: { hi: 'दस्तावेज जमा करें', en: 'Submit Documents' },
              description: { hi: 'आवश्यक दस्तावेज अपलोड करें', en: 'Upload required documents' }
            },
            {
              title: { hi: 'सत्यापन', en: 'Verification' },
              description: { hi: 'फील्ड वेरिफिकेशन होगा', en: 'Field verification will be done' }
            }
          ]
        },
        active: true
      }
    }),

    prisma.scheme.create({
      data: {
        title: {
          hi: 'सोलर पंप योजना',
          en: 'Solar Pump Scheme'
        },
        description: {
          hi: 'किसानों के लिए सब्सिडी पर सोलर वॉटर पंप',
          en: 'Subsidized solar water pumps for farmers'
        },
        category: 'agriculture',
        criteriaJSON: {
          eligibility: [
            { field: 'occupation', operator: '==', value: 'farmer' },
            { field: 'landSize', operator: '>', value: 0.5 },
            { field: 'hasElectricityConnection', operator: '==', value: false }
          ]
        },
        docsRequired: ['किसान कार्ड', 'भूमि दस्तावेज', 'बैंक पासबुक'],
        active: true
      }
    })
  ]);

  console.log('✅ Sample schemes created');

  // Create sample forms
  const complaintForm = await prisma.form.create({
    data: {
      name: 'शिकायत पंजीकरण',
      slug: 'complaint-registration',
      schemaJSON: {
        type: 'object',
        properties: {
          complaintType: {
            type: 'string',
            enum: ['infrastructure', 'sanitation', 'water', 'electricity', 'other'],
            title: { hi: 'शिकायत का प्रकार', en: 'Complaint Type' }
          },
          description: {
            type: 'string',
            title: { hi: 'शिकायत का विवरण', en: 'Complaint Description' },
            minLength: 10
          },
          location: {
            type: 'string',
            title: { hi: 'स्थान', en: 'Location' }
          },
          contactPhone: {
            type: 'string',
            pattern: '^[6-9]\\d{9}$',
            title: { hi: 'संपर्क नंबर', en: 'Contact Number' }
          },
          priority: {
            type: 'string',
            enum: ['low', 'medium', 'high', 'urgent'],
            default: 'medium',
            title: { hi: 'प्राथमिकता', en: 'Priority' }
          }
        },
        required: ['complaintType', 'description', 'location', 'contactPhone']
      },
      slaDays: 7,
      workflowJSON: {
        assignmentRules: [
          {
            condition: { complaintType: 'infrastructure' },
            assignTo: 'infrateam@ummidsehari.in'
          },
          {
            condition: { complaintType: 'water' },
            assignTo: 'waterteam@ummidsehari.in'
          }
        ]
      },
      active: true
    }
  });

  console.log('✅ Complaint form created');

  // Create sample submissions
  const submissions = await Promise.all([
    prisma.submission.create({
      data: {
        formId: complaintForm.id,
        userId: null, // Anonymous submission
        dataJSON: {
          complaintType: 'water',
          description: 'मुख्य सड़क पर पानी का पाइप टूटा है और दिन भर पानी बह रहा है',
          location: 'मुख्य सड़क, दुकान के पास',
          contactPhone: '9876543210',
          priority: 'high'
        },
        status: 'IN_PROGRESS',
        geo: { lat: 28.6139, lng: 77.2090 },
        slaDue: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        history: [
          {
            status: 'PENDING',
            timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
            note: 'शिकायत दर्ज की गई'
          },
          {
            status: 'IN_PROGRESS',
            timestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
            note: 'टीम को असाइन की गई'
          }
        ]
      }
    })
  ]);

  console.log('✅ Sample submissions created');

  // Create sample events
  const events = await Promise.all([
    prisma.event.create({
      data: {
        title: {
          hi: 'ग्राम सभा बैठक',
          en: 'Gram Sabha Meeting'
        },
        description: {
          hi: 'मासिक ग्राम सभा की बैठक, सभी ग्रामवासी आमंत्रित हैं',
          en: 'Monthly Gram Sabha meeting, all villagers are invited'
        },
        startDate: new Date('2024-12-15T10:00:00Z'),
        endDate: new Date('2024-12-15T14:00:00Z'),
        location: 'पंचायत भवन',
        geo: {
          lat: 28.6139,
          lng: 77.2090
        },
        rsvpEnabled: true,
        maxAttendees: 200,
        tags: ['gram-sabha', 'meeting', 'governance']
      }
    }),

    prisma.event.create({
      data: {
        title: {
          hi: 'वृक्षारोपण कार्यक्रम',
          en: 'Tree Plantation Program'
        },
        description: {
          hi: 'सामुदायिक वृक्षारोपण कार्यक्रम - सभी उम्र के लोग भाग ले सकते हैं',
          en: 'Community tree plantation program - people of all ages can participate'
        },
        startDate: new Date('2024-12-20T08:00:00Z'),
        endDate: new Date('2024-12-20T12:00:00Z'),
        location: 'स्कूल के पीछे का मैदान',
        rsvpEnabled: true,
        maxAttendees: 100,
        tags: ['environment', 'trees', 'community']
      }
    })
  ]);

  console.log('✅ Sample events created');

  // Create sample pledges
  const pledges = await Promise.all([
    prisma.pledge.create({
      data: {
        userId: null,
        pledgeType: 'TREE',
        title: '50 पेड़ लगाने का संकल्प',
        amount: 50,
        description: 'अपने घर के आस-पास और खाली जमीन पर फलदार पेड़ लगाऊंगा',
        approved: true
      }
    }),
    prisma.pledge.create({
      data: {
        userId: editorUser.id,
        pledgeType: 'SOLAR',
        title: 'घर पर सोलर पैनल लगाना',
        amount: 3, // 3kW
        description: '3kW का सोलर पैनल लगाकर बिजली का बिल कम करना',
        approved: true
      }
    })
  ]);

  console.log('✅ Sample pledges created');

  // Create directory entries
  const directoryEntries = await Promise.all([
    prisma.directoryEntry.create({
      data: {
        type: 'SHG',
        name: 'माँ दुर्गा महिला स्वयं सहायता समूह',
        contact: {
          phone: '+91-9876543211',
          email: 'durga.shg@example.com',
          address: 'वार्ड-2, दामदय'
        },
        description: {
          hi: 'महिला सशक्तिकरण और आर्थिक स्वावलंबन के लिए काम करने वाला समूह',
          en: 'Group working for women empowerment and economic self-reliance'
        },
        products: [
          {
            name: { hi: 'हस्तनिर्मित साबुन', en: 'Handmade Soap' },
            price: 25,
            unit: 'piece'
          },
          {
            name: { hi: 'आचार-मुरब्बा', en: 'Pickles & Preserves' },
            price: 150,
            unit: 'kg'
          }
        ],
        tags: ['women', 'self-help', 'handicrafts'],
        approved: true,
        featured: true
      }
    }),

    prisma.directoryEntry.create({
      data: {
        type: 'BUSINESS',
        name: 'रामजी किराना स्टोर',
        contact: {
          phone: '+91-9876543212',
          address: 'मुख्य बाजार, दामदय'
        },
        description: {
          hi: 'दैनिक उपयोग की सभी वस्तुएं उपलब्ध',
          en: 'All daily use items available'
        },
        products: [
          {
            name: { hi: 'किराना सामान', en: 'Grocery Items' },
            description: { hi: 'दाल, चावल, तेल, मसाले', en: 'Pulses, rice, oil, spices' }
          }
        ],
        tags: ['grocery', 'daily-needs'],
        approved: true
      }
    })
  ]);

  console.log('✅ Directory entries created');

  // Create translation keys for UI
  const translationKeys = await Promise.all([
    prisma.translationKey.create({
      data: {
        key: 'common.welcome',
        defaultText: 'Welcome',
        module: 'common',
        translations: {
          create: [
            { locale: 'hi-IN', text: 'स्वागत है' },
            { locale: 'en-IN', text: 'Welcome' }
          ]
        }
      }
    }),
    prisma.translationKey.create({
      data: {
        key: 'nav.services',
        defaultText: 'Services',
        module: 'navigation',
        translations: {
          create: [
            { locale: 'hi-IN', text: 'सेवाएं' },
            { locale: 'en-IN', text: 'Services' }
          ]
        }
      }
    }),
    prisma.translationKey.create({
      data: {
        key: 'nav.projects',
        defaultText: 'Projects',
        module: 'navigation',
        translations: {
          create: [
            { locale: 'hi-IN', text: 'परियोजनाएं' },
            { locale: 'en-IN', text: 'Projects' }
          ]
        }
      }
    })
  ]);

  console.log('✅ Translation keys created');

  // Create feature flags
  const featureFlags = await Promise.all([
    prisma.featureFlag.create({
      data: {
        key: 'enable_pwa_install_prompt',
        enabled: true,
        description: 'Show PWA installation prompt to users'
      }
    }),
    prisma.featureFlag.create({
      data: {
        key: 'enable_background_sync',
        enabled: true,
        description: 'Enable background synchronization for offline forms'
      }
    }),
    prisma.featureFlag.create({
      data: {
        key: 'enable_advanced_analytics',
        enabled: false,
        description: 'Enable advanced analytics and tracking'
      }
    })
  ]);

  console.log('✅ Feature flags created');

  // Create audit log for seeding
  await prisma.auditLog.create({
    data: {
      actorId: adminUser.id,
      action: 'seed',
      entity: 'Database',
      diffJSON: {
        event: 'database_seeded',
        timestamp: new Date().toISOString(),
        records_created: {
          users: 3,
          projects: 2,
          schemes: 2,
          forms: 1,
          submissions: 1,
          events: 2,
          pledges: 2,
          directory_entries: 2,
          translation_keys: 3,
          feature_flags: 3
        }
      }
    }
  });

  console.log('✅ Audit log entry created');

  console.log(`
🎉 Database seeding completed successfully!

👤 Admin User Created:
   Email: admin@ummidsehari.in
   2FA Secret: ${adminSecret.base32}
   2FA QR Code URL: ${speakeasy.otpauthURL({ secret: adminSecret.base32, label: 'admin@ummidsehari.in', name: 'Ummid Se Hari', issuer: 'Damday GP' })}

📊 Sample Data Created:
   - 3 Users (admin, editor, data-entry)  
   - 2 Projects with milestones
   - 2 Government schemes with eligibility criteria
   - 1 Complaint form with sample submission
   - 2 Upcoming events with RSVP
   - 2 Pledges for tree planting and solar
   - 2 Directory entries (SHG and business)
   - Translation keys for Hindi/English UI
   - Feature flags for PWA and advanced features

🚀 Ready for advanced development!
  `);
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error('❌ Seeding failed:', e);
    await prisma.$disconnect();
    process.exit(1);
  });
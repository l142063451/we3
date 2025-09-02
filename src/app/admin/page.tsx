'use client'

import React from 'react'
import { useSession } from 'next-auth/react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Users, 
  FileText, 
  Settings, 
  BarChart3, 
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Activity
} from 'lucide-react'

export default function AdminDashboard() {
  const { data: session } = useSession()

  // Mock data - will be replaced with real API calls
  const stats = [
    {
      title: 'कुल उपयोगकर्ता',
      value: '1,234',
      change: '+12%',
      icon: Users,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100'
    },
    {
      title: 'सक्रिय परियोजनाएं',
      value: '45',
      change: '+8%', 
      icon: FileText,
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    },
    {
      title: 'लंबित अनुरोध',
      value: '23',
      change: '-5%',
      icon: Clock,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100'
    },
    {
      title: 'सिस्टम स्वास्थ्य',
      value: '98%',
      change: '+1%',
      icon: Activity,
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    }
  ]

  const recentActivities = [
    {
      id: 1,
      user: 'राज कुमार',
      action: 'नई शिकायत दर्ज की',
      time: '2 मिनट पहले',
      type: 'complaint',
      status: 'pending'
    },
    {
      id: 2,
      user: 'सुनीता देवी',
      action: 'योजना के लिए आवेदन किया',
      time: '15 मिनट पहले',
      type: 'application',
      status: 'processing'
    },
    {
      id: 3,
      user: 'व्यवस्थापक',
      action: 'नई परियोजना अपरूवल',
      time: '1 घंटे पहले',
      type: 'approval',
      status: 'completed'
    },
    {
      id: 4,
      user: 'अमित शर्मा',
      action: 'डेटा अपडेट किया',
      time: '2 घंटे पहले',
      type: 'update',
      status: 'completed'
    }
  ]

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge variant="secondary"><Clock className="w-3 h-3 mr-1" />लंबित</Badge>
      case 'processing':
        return <Badge variant="default"><AlertTriangle className="w-3 h-3 mr-1" />प्रसंस्करण</Badge>
      case 'completed':
        return <Badge variant="outline" className="text-green-600"><CheckCircle className="w-3 h-3 mr-1" />पूर्ण</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2">
          स्वागत है, {session?.user?.name || 'व्यवस्थापक'}
        </h1>
        <p className="text-blue-100">
          उम्मीद से हरी प्रशासन पैनल में आपका स्वागत है। आज की गतिविधियों का अवलोकन करें।
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <Card key={stat.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <div className={`p-2 rounded-full ${stat.bgColor}`}>
                <stat.icon className={`h-4 w-4 ${stat.color}`} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                <span className={stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'}>
                  {stat.change}
                </span>{' '}
                पिछले महीने से
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Activities */}
        <Card>
          <CardHeader>
            <CardTitle>हाल की गतिविधियां</CardTitle>
            <CardDescription>
              सिस्टम में हाल की उपयोगकर्ता गतिविधियों का अवलोकन
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivities.map((activity) => (
                <div key={activity.id} className="flex items-center space-x-4 p-3 rounded-lg border">
                  <div className="flex-1 space-y-1">
                    <p className="text-sm font-medium leading-none">
                      {activity.user}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {activity.action}
                    </p>
                  </div>
                  <div className="flex flex-col items-end space-y-1">
                    {getStatusBadge(activity.status)}
                    <p className="text-xs text-muted-foreground">
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>त्वरित कार्य</CardTitle>
            <CardDescription>
              सामान्य प्रशासनिक कार्यों के लिए शॉर्टकट
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <button className="flex flex-col items-center p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <Users className="h-8 w-8 text-blue-600 mb-2" />
                <span className="text-sm font-medium">उपयोगकर्ता प्रबंधन</span>
              </button>
              
              <button className="flex flex-col items-center p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <FileText className="h-8 w-8 text-green-600 mb-2" />
                <span className="text-sm font-medium">सामग्री प्रबंधन</span>
              </button>
              
              <button className="flex flex-col items-center p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <BarChart3 className="h-8 w-8 text-purple-600 mb-2" />
                <span className="text-sm font-medium">रिपोर्ट</span>
              </button>
              
              <button className="flex flex-col items-center p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <Settings className="h-8 w-8 text-gray-600 mb-2" />
                <span className="text-sm font-medium">सेटिंग्स</span>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Security & System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="h-5 w-5 mr-2" />
              सुरक्षा स्थिति
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">2FA सक्रिय उपयोगकर्ता</span>
                <Badge variant="outline" className="text-green-600">87%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">असफल लॉगिन प्रयास</span>
                <Badge variant="secondary">3 आज</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">SSL प्रमाणपत्र</span>
                <Badge variant="outline" className="text-green-600">मान्य</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              सिस्टम मॉनिटरिंग
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">सर्वर अपटाइम</span>
                <Badge variant="outline" className="text-green-600">99.9%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">डेटाबेस प्रदर्शन</span>
                <Badge variant="outline" className="text-green-600">अच्छा</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">बैकअप स्थिति</span>
                <Badge variant="outline" className="text-green-600">नवीनतम</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
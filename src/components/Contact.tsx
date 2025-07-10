
import React from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { 
  Mail, 
  MessageSquare, 
  Phone, 
  Calendar, 
  MapPin, 
  Linkedin, 
  Github, 
  Clock,
  CheckCircle
} from 'lucide-react';

const Contact = () => {
  return (
    <section id="contact" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Let's Automate Together
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Ready to transform your business processes? Let's discuss how automation can solve your 
            specific challenges and drive efficiency across your organization.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <Card className="border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="text-2xl text-gray-900">Send a Message</CardTitle>
              <CardDescription className="text-gray-600">
                Tell me about your automation needs and challenges. I'll get back to you within 24 hours.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    First Name
                  </label>
                  <Input placeholder="John" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Last Name
                  </label>
                  <Input placeholder="Doe" />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email Address
                </label>
                <Input type="email" placeholder="john@company.com" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Company
                </label>
                <Input placeholder="Your Company Name" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Project Type
                </label>
                <select className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                  <option>Select your project type</option>
                  <option>Power Automate Workflow</option>
                  <option>Excel Analytics & Dashboards</option>
                  <option>SharePoint Automation</option>
                  <option>API Integration</option>
                  <option>WhatsApp Business Integration</option>
                  <option>Complete Process Automation</option>
                  <option>Consultation Only</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Project Details
                </label>
                <Textarea 
                  placeholder="Describe your current processes, challenges, and what you'd like to automate..."
                  rows={4}
                />
              </div>
              
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-lg py-3">
                Send Message
                <Mail className="ml-2 h-5 w-5" />
              </Button>
            </CardContent>
          </Card>

          {/* Contact Info & Quick Actions */}
          <div className="space-y-8">
            {/* Quick Contact Options */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Button size="lg" className="bg-green-600 hover:bg-green-700 h-auto py-4 px-6">
                <MessageSquare className="mr-3 h-6 w-6" />
                <div className="text-left">
                  <div className="font-semibold">WhatsApp</div>
                  <div className="text-sm opacity-90">Quick consultation</div>
                </div>
              </Button>
              
              <Button size="lg" variant="outline" className="h-auto py-4 px-6 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white">
                <Calendar className="mr-3 h-6 w-6" />
                <div className="text-left">
                  <div className="font-semibold">Book Call</div>
                  <div className="text-sm opacity-70">Free consultation</div>
                </div>
              </Button>
            </div>

            {/* Contact Information */}
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6 space-y-6">
                <h3 className="text-xl font-bold text-gray-900">Get In Touch</h3>
                
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-blue-100 rounded-lg">
                      <Mail className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Email</p>
                      <p className="text-gray-600">contact@automationexpert.com</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-green-100 rounded-lg">
                      <Phone className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Phone</p>
                      <p className="text-gray-600">+1 (555) 123-4567</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-purple-100 rounded-lg">
                      <Clock className="h-5 w-5 text-purple-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Response Time</p>
                      <p className="text-gray-600">Within 24 hours</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="p-3 bg-gray-100 rounded-lg">
                      <MapPin className="h-5 w-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Location</p>
                      <p className="text-gray-600">Remote & On-site available</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Social Links */}
            <Card className="border-0 shadow-lg">
              <CardContent className="p-6">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Connect With Me</h3>
                <div className="flex gap-4">
                  <Button variant="outline" size="lg" className="flex-1">
                    <Linkedin className="mr-2 h-5 w-5" />
                    LinkedIn
                  </Button>
                  <Button variant="outline" size="lg" className="flex-1">
                    <Github className="mr-2 h-5 w-5" />
                    GitHub
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Service Guarantee */}
            <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-50 to-indigo-50">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="p-2 bg-blue-600 rounded-lg">
                    <CheckCircle className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="font-bold text-gray-900 mb-2">Service Guarantee</h3>
                    <ul className="space-y-1 text-sm text-gray-600">
                      <li>• Free initial consultation</li>
                      <li>• 30-day support included</li>
                      <li>• Money-back guarantee</li>
                      <li>• Complete documentation provided</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;

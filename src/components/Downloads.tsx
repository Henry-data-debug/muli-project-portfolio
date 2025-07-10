
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, FileText, FileSpreadsheet, Workflow, Star, Users } from 'lucide-react';

const Downloads = () => {
  const downloads = [
    {
      title: "Invoice Processing Power Automate Flow",
      description: "Complete workflow template that handles invoice receipt, data extraction, validation, and approval routing with AI Builder integration.",
      type: "Power Automate Template",
      category: "Workflow",
      icon: Workflow,
      downloads: "2.5k",
      rating: 4.9,
      features: ["AI-powered data extraction", "Multi-stage approval", "Error handling", "SharePoint integration"],
      fileSize: "1.2 MB"
    },
    {
      title: "Advanced Excel Financial Dashboard",
      description: "Dynamic financial dashboard with KPI tracking, scenario analysis, and automated data visualization using advanced formulas and pivot tables.",
      type: "Excel Template",
      category: "Analytics",
      icon: FileSpreadsheet,
      downloads: "3.1k",
      rating: 4.8,
      features: ["Dynamic charts", "Scenario modeling", "KPI tracking", "Data validation"],
      fileSize: "2.8 MB"
    },
    {
      title: "SharePoint Document Approval Workflow",
      description: "Automated document review and approval system with version control, stakeholder notifications, and deadline management.",
      type: "SharePoint Solution",
      category: "Document Management",
      icon: FileText,
      downloads: "1.8k",
      rating: 4.7,
      features: ["Version control", "Auto notifications", "Deadline tracking", "Mobile friendly"],
      fileSize: "850 KB"
    },
    {
      title: "WhatsApp Business API Integration Guide",
      description: "Complete setup guide with code samples, authentication flows, and best practices for implementing WhatsApp Business API in your applications.",
      type: "Documentation + Code",
      category: "API Integration",
      icon: FileText,
      downloads: "4.2k",
      rating: 4.9,
      features: ["Step-by-step guide", "Code examples", "Authentication setup", "Error handling"],
      fileSize: "3.5 MB"
    },
    {
      title: "Power BI Dashboard Templates Bundle",
      description: "Collection of 12 professional dashboard templates for sales, finance, operations, and HR with dynamic data connections and interactive visuals.",
      type: "Power BI Templates",
      category: "Business Intelligence",
      icon: FileSpreadsheet,
      downloads: "5.7k",
      rating: 4.8,
      features: ["12 dashboard templates", "Interactive visuals", "Dynamic connections", "Mobile optimized"],
      fileSize: "15.4 MB"
    },
    {
      title: "Workflow Automation Checklist",
      description: "Comprehensive checklist and assessment tool to identify automation opportunities in your business processes with ROI calculations.",
      type: "PDF Guide",
      category: "Strategy",
      icon: FileText,
      downloads: "6.8k",
      rating: 4.6,
      features: ["Process assessment", "ROI calculator", "Implementation roadmap", "Best practices"],
      fileSize: "2.1 MB"
    }
  ];

  return (
    <section className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Free Downloads & Templates
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Ready-to-use templates, workflows, and guides to jumpstart your automation journey. 
            All downloads include documentation and setup instructions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {downloads.map((item, index) => (
            <Card key={index} className="border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 h-full flex flex-col">
              <CardHeader className="pb-4">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <item.icon className="h-6 w-6 text-blue-600" />
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {item.category}
                  </Badge>
                </div>
                
                <CardTitle className="text-lg font-bold text-gray-900 leading-tight">
                  {item.title}
                </CardTitle>
                <CardDescription className="text-gray-600 leading-relaxed">
                  {item.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="flex-grow flex flex-col">
                <div className="mb-4">
                  <p className="text-sm font-medium text-blue-600 mb-2">{item.type}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      {item.rating}
                    </div>
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4" />
                      {item.downloads} downloads
                    </div>
                  </div>
                </div>
                
                <div className="mb-6 flex-grow">
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Features:</h4>
                  <ul className="space-y-1">
                    {item.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="text-sm text-gray-600 flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-blue-600 rounded-full"></div>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="border-t pt-4 mt-auto">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-600">File size: {item.fileSize}</span>
                    <Badge variant="secondary" className="bg-green-100 text-green-800">
                      Free
                    </Badge>
                  </div>
                  <Button className="w-full bg-blue-600 hover:bg-blue-700">
                    <Download className="mr-2 h-4 w-4" />
                    Download Now
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="bg-white rounded-2xl p-8 shadow-lg max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Want Custom Templates?
            </h3>
            <p className="text-gray-600 mb-6">
              Need a specific workflow or template tailored to your business needs? 
              I create custom automation solutions that fit your exact requirements.
            </p>
            <Button size="lg" className="bg-green-600 hover:bg-green-700">
              Request Custom Template
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Downloads;

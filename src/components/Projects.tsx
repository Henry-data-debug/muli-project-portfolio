
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, TrendingUp, Users, Clock, CheckCircle } from 'lucide-react';

const Projects = () => {
  const projects = [
    {
      title: "Automated Invoice Processing System",
      description: "Built a comprehensive Power Automate workflow that processes incoming invoices, extracts data using AI, validates against purchase orders, and routes for approval - reducing processing time by 85%.",
      image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?q=80&w=600&h=400&fit=crop",
      tags: ["Power Automate", "SharePoint", "AI Builder", "Excel"],
      metrics: [
        { label: "Time Saved", value: "85%", icon: Clock },
        { label: "Processing Speed", value: "10x", icon: TrendingUp },
        { label: "Accuracy Rate", value: "99.8%", icon: CheckCircle }
      ],
      category: "Workflow Automation"
    },
    {
      title: "WhatsApp Business Integration",
      description: "Developed an automated customer support system using WhatsApp Business API that handles inquiries, schedules appointments, and sends order updates - serving 1000+ customers monthly.",
      image: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=600&h=400&fit=crop",
      tags: ["WhatsApp API", "Power Automate", "Customer Service", "Automation"],
      metrics: [
        { label: "Customers Served", value: "1000+", icon: Users },
        { label: "Response Time", value: "<30s", icon: Clock },
        { label: "Satisfaction Rate", value: "94%", icon: TrendingUp }
      ],
      category: "API Integration"
    },
    {
      title: "Dynamic Business Dashboard",
      description: "Created an interactive Power BI dashboard with real-time KPI tracking, automated data refresh from multiple sources, and alert system for critical metrics - used by C-suite executives.",
      image: "https://images.unsplash.com/photo-1487058792275-0ad4aaf24ca7?q=80&w=600&h=400&fit=crop",
      tags: ["Power BI", "Excel", "Data Analysis", "KPI Tracking"],
      metrics: [
        { label: "Data Sources", value: "15+", icon: CheckCircle },
        { label: "Daily Users", value: "50+", icon: Users },
        { label: "Update Frequency", value: "Real-time", icon: TrendingUp }
      ],
      category: "Business Intelligence"
    },
    {
      title: "SharePoint Document Workflow",
      description: "Implemented an automated document approval system with version control, stakeholder notifications, and deadline tracking - managing 500+ documents monthly with zero bottlenecks.",
      image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?q=80&w=600&h=400&fit=crop",
      tags: ["SharePoint", "Power Automate", "Document Management", "Workflows"],
      metrics: [
        { label: "Documents Processed", value: "500+", icon: CheckCircle },
        { label: "Approval Time", value: "60% faster", icon: Clock },
        { label: "Error Reduction", value: "95%", icon: TrendingUp }
      ],
      category: "Document Management"
    },
    {
      title: "Advanced Excel Analytics Suite",
      description: "Developed complex financial models with dynamic formulas, scenario analysis, and automated reporting features - supporting strategic decision making for multiple departments.",
      image: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?q=80&w=600&h=400&fit=crop",
      tags: ["Excel", "VBA", "Financial Modeling", "Analytics"],
      metrics: [
        { label: "Formula Complexity", value: "Advanced", icon: TrendingUp },
        { label: "Departments Served", value: "8", icon: Users },
        { label: "Analysis Speed", value: "10x faster", icon: Clock }
      ],
      category: "Data Analytics"
    },
    {
      title: "Multi-Platform Integration Hub",
      description: "Built a centralized automation hub connecting CRM, ERP, email marketing, and social media platforms using Zapier and custom APIs - synchronizing data across 12 different systems.",
      image: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=600&h=400&fit=crop",
      tags: ["Zapier", "API Integration", "CRM", "ERP"],
      metrics: [
        { label: "Systems Connected", value: "12", icon: CheckCircle },
        { label: "Data Sync Rate", value: "99.9%", icon: TrendingUp },
        { label: "Manual Work Reduced", value: "90%", icon: Clock }
      ],
      category: "System Integration"
    }
  ];

  return (
    <section id="projects" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Projects & Case Studies
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Real-world automation solutions that deliver measurable results and transform business operations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <Card key={index} className="overflow-hidden hover:shadow-xl transition-all duration-300 hover:-translate-y-1 border-0 shadow-lg">
              <div className="relative">
                <img 
                  src={project.image} 
                  alt={project.title}
                  className="w-full h-48 object-cover"
                />
                <div className="absolute top-4 left-4">
                  <Badge variant="secondary" className="bg-white/90 text-gray-800 font-medium">
                    {project.category}
                  </Badge>
                </div>
              </div>
              
              <CardHeader className="pb-4">
                <CardTitle className="text-xl font-bold text-gray-900 leading-tight">
                  {project.title}
                </CardTitle>
                <CardDescription className="text-gray-600 leading-relaxed">
                  {project.description}
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag, tagIndex) => (
                    <Badge key={tagIndex} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  {project.metrics.map((metric, metricIndex) => (
                    <div key={metricIndex} className="text-center p-3 bg-gray-50 rounded-lg">
                      <metric.icon className="h-5 w-5 text-blue-600 mx-auto mb-1" />
                      <div className="font-bold text-gray-900 text-sm">{metric.value}</div>
                      <div className="text-xs text-gray-600">{metric.label}</div>
                    </div>
                  ))}
                </div>
                
                <Button variant="outline" className="w-full group">
                  View Case Study Details
                  <ExternalLink className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
            View All Projects
            <ExternalLink className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </section>
  );
};

export default Projects;

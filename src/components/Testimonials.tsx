
import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Star, Quote } from 'lucide-react';

const Testimonials = () => {
  const testimonials = [
    {
      name: "Grace Wanjiku",
      position: "Operations Manager",
      company: "Nairobi Tech Hub",
      content: "Henry's automation workflows transformed our entire invoice processing system. What used to take our team 3 days now happens in 30 minutes. The ROI was immediate and the reliability is outstanding.",
      rating: 5
    },
    {
      name: "Peter Kamau",
      position: "CEO",
      company: "DataLink Kenya",
      content: "The WhatsApp integration revolutionized our customer service. We now handle 10x more inquiries with the same team size. The automated responses are so natural, customers love the instant support.",
      rating: 5
    },
    {
      name: "Anne Njeri",
      position: "Finance Director",
      company: "GrowthCorp East Africa",
      content: "The Power BI dashboards provide real-time insights that drive our strategic decisions. The automated reporting saves us 20 hours per week and the accuracy is perfect every time.",
      rating: 5
    },
    {
      name: "Samuel Kipchoge",
      position: "IT Manager",
      company: "InnovateTech Nairobi",
      content: "The SharePoint automation eliminated our document bottlenecks completely. Approval workflows that took weeks now complete in days, and we have full visibility into every step of the process.",
      rating: 5
    },
    {
      name: "Mary Akinyi",
      position: "Marketing Director",
      company: "BrandBuilders Kenya",
      content: "Henry's multi-platform integration connected all our marketing tools seamlessly. Campaign data flows automatically between systems, giving us insights we never had before. Game-changing work!",
      rating: 5
    },
    {
      name: "John Mutua",
      position: "COO",
      company: "EfficiencyFirst Ltd",
      content: "The Excel analytics suite Henry built handles complex financial modeling that used to require expensive software. The automation features save our analysts hours every day while improving accuracy.",
      rating: 5
    }
  ];

  return (
    <section className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Client Testimonials
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Hear from Kenyan business leaders who have transformed their operations through intelligent automation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
              <CardContent className="p-6 h-full flex flex-col">
                <div className="flex items-center gap-1 mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="h-5 w-5 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                
                <div className="relative mb-6 flex-grow">
                  <Quote className="h-8 w-8 text-blue-600/20 absolute -top-2 -left-1" />
                  <p className="text-gray-600 leading-relaxed pl-6">
                    "{testimonial.content}"
                  </p>
                </div>
                
                <div className="flex items-center gap-4 mt-auto">
                  <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
                    <span className="text-blue-600 font-semibold text-lg">
                      {testimonial.name.split(' ').map(n => n[0]).join('')}
                    </span>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">{testimonial.name}</h4>
                    <p className="text-sm text-gray-600">{testimonial.position}</p>
                    <p className="text-sm text-blue-600 font-medium">{testimonial.company}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex items-center gap-8 bg-white rounded-2xl px-8 py-6 shadow-lg">
            <div className="text-center">
              <div className="flex items-center justify-center gap-1 mb-2">
                {[...Array(5)].map((_, i) => (
                  <Star key={i} className="h-5 w-5 fill-yellow-400 text-yellow-400" />
                ))}
              </div>
              <div className="text-2xl font-bold text-gray-900">4.9/5</div>
              <div className="text-sm text-gray-600">Average Rating</div>
            </div>
            <div className="w-px h-12 bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">98%</div>
              <div className="text-sm text-gray-600">Client Satisfaction</div>
            </div>
            <div className="w-px h-12 bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">100+</div>
              <div className="text-sm text-gray-600">Success Stories</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonials;

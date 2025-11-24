import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";

export const metadata: Metadata = {
    title: "Gazoo Analyst",
    description: "AI-Powered Race Analysis",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="bg-[#191818] text-white min-h-screen">
                <Header />
                <main className="p-6">
                    {children}
                </main>
            </body>
        </html>
    );
}

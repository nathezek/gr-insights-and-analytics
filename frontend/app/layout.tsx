import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

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
            <body className="flex bg-[#191818] text-white min-h-screen">
                <Sidebar />
                <main className="flex-1 overflow-auto p-8">
                    {children}
                </main>
            </body>
        </html>
    );
}
